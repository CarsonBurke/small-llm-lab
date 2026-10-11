"""GPU causality of the hierarchical byte model (queue through mlq).

The chunked KDA kernel is causal in exact arithmetic but rescales within a
64-token chunk by the gate cumsum, so a later patch perturbs an earlier
patch's output at rounding level (~1e-4 in fp32; the token reference shares
this). Exact causality of everything else -- hashing, encoder, patcher,
routing, decoders -- is therefore checked with an attention-only trunk, and
the KDA trunk against a noise bound.

The NextLat decoder has the same kind of rounding path: it steps only the
patches still open at each level, so a later patch's length changes the
GEMM row count, and cuBLAS's kernel choice, for earlier ones (~2^-9 nats in
bf16). Its exact causality is checked on a fixed-shape reference that steps
every route through every level, and the fast decoder against that reference.
"""

from __future__ import annotations

import re
import types

import pytest
import torch

from pretraining.hbyte.model import (
    HashConfig,
    LocalConfig,
    ModelConfig,
    NextLatDecoder,
    TrunkConfig,
    build_model,
    initialize,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VARIANTS = [
    ("static", "attention"),
    ("static", "nextlat"),
    ("space", "attention"),
    ("entropy", "attention"),
    ("entropy", "nextlat"),
    ("learned", "attention"),
    ("learned", "nextlat"),
]
# Optional pieces, exercised on the learned-patching NextLat model.
EXTRAS = {"nextlat_local_reset": True, "patch_pool": "attn"}
# The chunk architecture: lookups-only embedder, attention pooling into the
# trunk, NextLat unembedder, high length cap.
CHUNK = {"patch_pool": "attn", "encoder_layers": 0, "max_patch": 64}
# Direct trunk logits ride on the attention decoder.
LOGITS = {"trunk_logits": True}
OPTIONAL = [
    ("entropy", "nextlat", EXTRAS),
    ("entropy", "nextlat", CHUNK),
    ("learned", "nextlat", CHUNK),
    ("space", "attention", {**LOGITS, "max_patch": 16}),
    ("learned", "attention", LOGITS),
    # Block-anchored prefix memory on each kind of layout.
    ("static", "attention", {"prefix_bits": 10}),
    ("space", "attention", {"prefix_bits": 10, "max_patch": 16}),
    ("space", "attention", {"prefix_bits": 10, "max_patch": 16, "block_ngram": 3, "prefix_context": True}),
    ("space", "attention", {"prefix_bits": 10, "max_patch": 16, "block_ngram": 2, "memory_gate": True}),
    ("space", "attention", {"prefix_bits": 10, "max_patch": 16, "prefix_local": True}),
    # A narrower encoder window than the decoder's.
    ("space", "attention", {"prefix_bits": 10, "max_patch": 16, "encoder_window": 32}),
    ("learned", "attention", {"prefix_bits": 10, "trunk_logits": True}),
]
# KDA local layers (H-Net style), alone and mixed with window attention. The
# chunked KDA kernel is causal only up to rounding, like the KDA trunk.
KDA_LOCAL = [
    ("space", "attention", {"max_patch": 16, "encoder_mixers": "k", "decoder_mixers": "ka", "prefix_bits": 10}),
    ("static", "attention", {"decoder_mixers": "kk"}),
]


def _model(patcher: str, decoder: str, kda_layers: tuple[int, ...], **extra):
    encoder_layers = extra.pop("encoder_layers", 1)  # **extra is already a copy
    mixers = {k: extra.pop(k) for k in ("encoder_mixers", "decoder_mixers", "encoder_window") if k in extra}
    config = ModelConfig(
        atom="byte",
        seq_len=1024,
        trunk=TrunkConfig(kda_layers=kda_layers),
        local=LocalConfig(dim=256, window=128, encoder_layers=encoder_layers, **mixers),
        hashing=HashConfig(orders=(3, 6), buckets=4096),
        patch_size=4,
        patcher=patcher,
        decoder=decoder,
        **{"max_patch": 8, **extra},
    )
    model = build_model(config).cuda()
    initialize(model, 0)
    # Make the zero-initialized output branches live so the test has teeth.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and p.abs().sum() == 0:
                p.normal_(std=0.02)
            elif re.search(r"_gain(\.\d+)?$", name):  # zero-init gates of the prefix memory
                p.fill_(0.5)
    model.prepare("cuda")
    x = torch.randint(0, 256, (2, 1025), device="cuda")
    if model.patcher is not None:
        # One training forward sets the entropy threshold from this batch, so
        # the checked batch has natural (data-dependent) boundaries.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x[:, :-1], x[:, 1:])
        model.zero_grad(set_to_none=True)
        assert torch.isfinite(model.patcher.threshold)
    model.eval()
    return model, x


def _scores(model, inputs, targets, **kwargs):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(inputs, targets, **kwargs)[0]


def _dense_nextlat_forward(self, h_enc, trunk_out, route_index, offset, targets, confidence=None):
    """`NextLatDecoder.forward` with every route stepped through all ``cap``
    levels: shapes depend only on the row length, so row results do not
    depend on other patches' lengths."""
    B, T, _ = h_enc.shape
    g = torch.cat([self.bos.type_as(trunk_out).expand(B, 1, -1), trunk_out], dim=1)
    k1 = g.size(1)
    rows = torch.arange(B, device=h_enc.device)[:, None]
    t = torch.arange(T, device=h_enc.device).expand(B, -1)
    base = torch.full((B, k1 + 1), -1, dtype=torch.long, device=h_enc.device)
    base = base.scatter(1, torch.where(offset == 0, route_index, k1), t)[:, :k1]
    lifted = self.start(g)
    state = lifted
    if self.reset is not None:
        state = state + self.reset(h_enc[rows, base.clamp(min=0)] * (base >= 0).unsqueeze(-1)).float()
    # Levels past a route's last byte read clamped bytes; nothing reads them.
    states = [state]
    for j in range(1, self.cap):
        states.append(self.step(states[-1], h_enc[rows, (base + j).clamp(max=T - 1)]))
    grid = torch.stack(states, 2)  # (B, routes, cap, d)
    trunk_part = lifted[rows, route_index] if confidence is not None else None
    return self.readout(grid[rows, route_index, offset], targets, confidence, trunk_part)


def _use_dense_nextlat(model) -> None:
    model.decoder.forward = types.MethodType(_dense_nextlat_forward, model.decoder)


@pytest.mark.parametrize("patcher,decoder,extra", [(*v, {}) for v in VARIANTS] + OPTIONAL)
def test_scores_are_exactly_causal_with_attention_trunk(patcher, decoder, extra):
    model, x = _model(patcher, decoder, kda_layers=(), **extra)
    if isinstance(model.decoder, NextLatDecoder):
        _use_dense_nextlat(model)
    inputs, targets = x[:, :-1], x[:, 1:]
    base = _scores(model, inputs, targets)
    if patcher == "entropy":
        lengths = torch.diff(model.last_layout.end_index[0][model.last_layout.patch_valid[0]])
        assert lengths.float().std() > 0, "entropy patches should vary in length"
    # Mid-patch, patch-end (s % 4 == 3) and early positions.
    ablated = _scores(model, inputs, targets, ablate_global=True)
    assert not torch.equal(base, ablated), "the trunk route must be live"
    for s in (3, 5, 300, 511, 777):
        y = inputs.clone()
        y[:, s] = (y[:, s] + 17) % 256
        # The knock-out diagnostic must be causal too.
        for kwargs, ref in (({}, base), ({"ablate_global": True}, ablated)):
            moved = _scores(model, y, targets, **kwargs)
            # Position t scores targets[t] from inputs <= t: t < s must not move.
            assert torch.equal(ref[:, :s], moved[:, :s]), (s, kwargs)
            assert (ref[:, s:] != moved[:, s:]).any(), (s, kwargs)


NEXTLAT = [(p, d, e) for p, d, e in [(*v, {}) for v in VARIANTS] + OPTIONAL if d == "nextlat"]


@pytest.mark.parametrize("patcher,decoder,extra", NEXTLAT)
def test_level_batched_nextlat_matches_dense_reference(patcher, decoder, extra):
    model, x = _model(patcher, decoder, kda_layers=(), **extra)
    inputs, targets = x[:, :-1], x[:, 1:]
    fast = _scores(model, inputs, targets)
    _use_dense_nextlat(model)
    diff = (fast - _scores(model, inputs, targets)).abs()
    # Same arithmetic up to GEMM kernel choice: bf16 rounding, not a bias.
    assert diff.max() < 0.02 and diff.mean() < 1e-3, (diff.max().item(), diff.mean().item())


@pytest.mark.parametrize("patcher,decoder,extra", [(*v, {}) for v in VARIANTS] + KDA_LOCAL)
def test_kda_future_influence_is_rounding_noise(patcher, decoder, extra):
    model, x = _model(patcher, decoder, kda_layers=TrunkConfig().kda_layers, **extra)
    inputs, targets = x[:, :-1], x[:, 1:]
    base = _scores(model, inputs, targets)
    for s in (5, 300, 777):
        y = inputs.clone()
        y[:, s] = (y[:, s] + 17) % 256
        moved = _scores(model, y, targets)
        past = (base[:, :s] - moved[:, :s]).abs().max().item()
        future = (base[:, s:] - moved[:, s:]).abs().max().item()
        assert past < 0.02, (s, past)
        assert future > 20 * past, (s, past, future)


@pytest.mark.parametrize("patcher,decoder,extra", [(*v, {}) for v in VARIANTS] + OPTIONAL + KDA_LOCAL)
def test_compiled_parts_match_eager_and_train(patcher, decoder, extra):
    model, x = _model(patcher, decoder, kda_layers=TrunkConfig().kda_layers, **extra)
    inputs, targets = x[:, :-1], x[:, 1:]
    eager = _scores(model, inputs, targets)
    eager_gate = model.last_gate
    model.compile_parts()
    compiled = _scores(model, inputs, targets)
    if model.patcher is None:  # boundaries fixed by the bytes alone
        assert (eager - compiled).abs().max() < 0.05
    else:
        # Compare the gate, not the realized ends: a rounding-level tie flip
        # re-routes every later byte and, once natural ends exceed the chunk
        # budget, shifts which later ones survive it.
        gate_error = (eager_gate - model.last_gate).abs().max()
        assert gate_error < 5e-3, gate_error
        threshold = model.patcher.threshold
        agreement = ((eager_gate > threshold) == (model.last_gate > threshold)).float().mean()
        assert agreement > 0.99, agreement
        assert (eager - compiled).abs().median() < 0.02
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        nll, aux = model(inputs, targets)
        loss = nll.sum() + (aux.sum() if aux is not None else 0)
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    missing = [n for n, g in grads.items() if g is None or not g.isfinite().all()]
    assert not missing, missing


@pytest.mark.parametrize("patcher,decoder", [("entropy", "attention"), ("learned", "nextlat")])
def test_varying_patch_count_does_not_recompile(patcher, decoder):
    from torch._dynamo.utils import counters

    model, x = _model(patcher, decoder, kda_layers=TrunkConfig().kda_layers)
    inputs, targets = x[:, :-1], x[:, 1:]
    model.compile_parts()
    base = float(model.patcher.threshold)
    _scores(model, inputs, targets)
    graphs = counters["stats"]["unique_graphs"]
    budget = model.config.chunk_budget(inputs.size(1))
    patch_counts = set()
    for scale in (0.5, 0.8, 1.2, 1.5):
        model.patcher.threshold.fill_(base * scale)
        _scores(model, inputs, targets)
        assert model.last_layout.end_index.size(1) == budget
        patch_counts.add(int(model.last_layout.patch_valid.sum()))
    assert len(patch_counts) > 1, "the thresholds should change the patch count"
    # NextLat's recurrence step is compiled with dynamic level sizes; it may
    # add at most its one dynamic graph.
    assert counters["stats"]["unique_graphs"] - graphs <= (decoder == "nextlat")


@pytest.mark.parametrize(
    "patcher,decoder,max_patch", [("entropy", "attention", 8), ("entropy", "nextlat", 64), ("learned", "nextlat", 64)]
)
def test_diagnostics_are_consistent(patcher, decoder, max_patch):
    from pretraining.hbyte.data import EvalSet
    from pretraining.hbyte.train import boundary_counterfactuals, evaluate

    model, _ = _model(patcher, decoder, kda_layers=TrunkConfig().kda_layers, max_patch=max_patch)
    model.compile_parts()
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (5, 1025), generator=gen)
    targets = x[:, 1:].clone()
    targets[:, :37] = -100  # unscored context, as in token-aligned rows
    eval_set = EvalSet("t", x[:, :-1].contiguous(), targets, int((targets >= 0).sum()))
    threshold = model.patcher.threshold.clone()
    plain = evaluate(model, eval_set, microbatch=2)
    out = evaluate(model, eval_set, microbatch=2, diagnostics=True)
    assert out["bpb"] == pytest.approx(plain["bpb"], rel=1e-6)
    # Every breakdown partitions the same codelength.
    gate = {"entropy": "gate_entropy_", "learned": "gate_p_"}[patcher]
    for prefix in ("class_", "ctx_", "offset_", gate):
        shares = {k[len(prefix):-len("_share")]: v for k, v in out.items() if k.startswith(prefix) and k.endswith("_share")}
        if prefix in ("offset_", gate):
            assert sum(shares.values()) == pytest.approx(1, abs=1e-9)
            mean = sum(out[f"{prefix}{name}_bpb"] * share for name, share in shares.items())
            assert mean == pytest.approx(out["bpb"], rel=1e-6), prefix
    assert -1 <= out[f"{gate}nll_corr"] <= 1
    assert 0 <= out["forced_end_fraction"] <= 1
    assert 0 <= out["dropped_end_fraction"] <= 1
    assert "|" in out["samples"]
    cf = boundary_counterfactuals(model, eval_set, 2, out, sweep=True)
    assert torch.equal(model.patcher.threshold, threshold)
    assert cf["random_bytes_per_patch"] == pytest.approx(out["bytes_per_patch"], rel=0.05)
    assert all(torch.isfinite(torch.tensor(v)) for v in cf.values())
    # A lower threshold ends more patches.
    assert cf["sweep_0.6_bytes_per_patch"] <= cf["sweep_1.5_bytes_per_patch"]


def test_block_prefix_ids_match_on_cuda():
    from pretraining.hbyte.model import block_prefix_ids, prefix_powers

    gen = torch.Generator().manual_seed(1)
    x = torch.randint(0, 257, (3, 4096), generator=gen)
    ends = torch.rand(3, 4096, generator=gen) < 0.2
    cpu = block_prefix_ids(x, ends, 18, prefix_powers(4096, "cpu"))
    cuda = block_prefix_ids(x.cuda(), ends.cuda(), 18, prefix_powers(4096, "cuda"))
    assert torch.equal(cpu, cuda.cpu())


def _fresh(**extra):
    config = ModelConfig(
        atom="byte",
        seq_len=1024,
        local=LocalConfig(dim=256, window=128),
        hashing=HashConfig(orders=(3, 6), buckets=4096),
        patcher="space",
        target_patch=6.0,
        max_patch=16,
        **extra,
    )
    model = build_model(config).cuda()
    initialize(model, 0)
    model.prepare("cuda")
    return model


def test_prefix_memory_starts_as_the_plain_model_and_learns():
    plain, memory = _fresh(), _fresh(
        prefix_bits=10, trunk_logits=True, block_ngram=3, prefix_context=True, memory_gate=True, prefix_local=True
    )
    shared = plain.state_dict()
    extra = memory.state_dict()
    assert all(torch.equal(v, extra[k]) for k, v in shared.items())
    x = torch.randint(0, 256, (2, 1025), device="cuda")
    inputs, targets = x[:, :-1], x[:, 1:]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.equal(plain(inputs, targets)[0], memory(inputs, targets)[0])
    # The byte head starts at zero, so nothing below it has gradient before
    # the first update; after it, the closed paths get gradient where they
    # open from.
    with torch.no_grad():
        memory.decoder.head.weight.normal_(std=0.02)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        memory(inputs, targets)[0].sum().backward()
    names = ("block_gain", "block_ngram_gain.0", "block_ngram_gain.1", "decoder.prefix_gain", "decoder.context_gain")
    for name in (*names, "decoder.prefix_head.weight", "decoder.trunk_head.weight"):
        grad = memory.get_parameter(name).grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_prefix_ids_are_wired_to_blocks_and_prefixes():
    from pretraining.hbyte.model import block_prefix_ids, spacelike_ends

    model = _fresh(prefix_bits=12)
    text = (b"the quick brown fox jumps over a lazy dog, again and again. " * 20)[:1025]
    x = torch.tensor([list(text)] * 2, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(x[:, :-1], x[:, 1:])
    layout = model.last_layout
    ends = layout.offset == 0
    assert torch.equal(ends, spacelike_ends(x[:, :-1]))  # no cap or budget ends in this text
    ids = block_prefix_ids(x[:, :-1], ends, 12, model.prefix_powers[1024])
    # Decoder: the open block's prefix, empty (0) right after an end.
    assert torch.equal(layout.prefix_id, torch.where(ends, 0, ids))
    # Trunk: each real slot carries its completed block's string.
    block_id = model.last_trunk_args[-1][..., 0]
    real = layout.patch_valid
    assert torch.equal(block_id[real], torch.gather(ids, 1, layout.end_index)[real])
    assert (block_id[real] > 0).all()
    # One id per block string, wherever the block sits.
    ids_of: dict[bytes, set[int]] = {}
    prev = -1
    for end, i in zip(layout.end_index[0][real[0]].tolist(), block_id[0][real[0]].tolist()):
        ids_of.setdefault(bytes(x[0, prev + 1 : end + 1].tolist()), set()).add(i)
        prev = end
    assert b" the " in ids_of and all(len(v) == 1 for v in ids_of.values())


def test_space_model_diagnostics():
    from pretraining.hbyte.data import EvalSet
    from pretraining.hbyte.train import evaluate

    model = _fresh(prefix_bits=10)
    model.compile_parts()
    model.eval()
    text = (b"Some words, then digits 12345 and more text.\n" * 100)[: 4 * 1025]
    x = torch.tensor(list(text)).view(4, 1025)
    targets = x[:, 1:].clone()
    targets[:, :20] = -100
    eval_set = EvalSet("t", x[:, :-1].contiguous(), targets, int((targets >= 0).sum()))
    out = evaluate(model, eval_set, microbatch=2, diagnostics=True)
    assert out["forced_end_fraction"] == pytest.approx(0, abs=0.05)
    assert out["dropped_end_fraction"] == 0
    shares = [v for k, v in out.items() if k.startswith("offset_") and k.endswith("_share")]
    assert sum(shares) == pytest.approx(1, abs=1e-9)

"""CPU contracts of the latent-feedback nanogpt-mini variants (tiny fp32 models)."""

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_feedback_model import (
    DEFAULT_DECAYS, FeedbackGPT, MemoryState, decayed_key_sum, decay_mask, linear_memory_features, linear_memory_read_dense, linear_memory_read_parallel, window_mask,
)
from pretraining.nanogpt_mini.nanogpt_mini_model import GPT

VOCAB, LAYERS, DIM, HEAD = 32, 2, 256, 128  # the trunk hardcodes 128-dim heads: DIM 256 gives 2 heads
DECAYS = (0.5, 0.95)


def _init(model: torch.nn.Module, seed: int) -> None:
    """The trainer's seeded init rules (proj zero, embed normal, matrices scaled normal, biases zero)."""
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        elif name in ("decay_logit", "temperature"):
            pass
        else:
            raise AssertionError(name)
    if hasattr(model, "reset_extra_parameters"):
        model.reset_extra_parameters()


def _model(mode: str, seed: int = 1, num_layers: int = LAYERS, **kwargs) -> FeedbackGPT:
    torch.manual_seed(seed)
    model = FeedbackGPT(VOCAB, num_layers, DIM, mode=mode, memory_head_dim=HEAD, decays=DECAYS, **kwargs).float()
    _init(model, seed)
    return model.eval()


def _randomize_projections(model: FeedbackGPT, seed: int = 7) -> None:
    """Make every zero-initialised map nonzero so the feedback paths are live."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "proj" in name and p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)


def _batch(seed: int = 3, B: int = 2, T: int = 8):
    g = torch.Generator().manual_seed(seed)
    inputs = torch.randint(0, VOCAB, (B, T), generator=g)
    targets = torch.randint(0, VOCAB, (B, T), generator=g)
    return inputs, targets


def test_none_mode_is_the_baseline_bitwise_including_seeded_init():
    torch.manual_seed(1)
    base = GPT(VOCAB, LAYERS, DIM).float()
    _init(base, 5)
    model = _model("none", seed=5)
    assert [n for n, _ in base.named_parameters()] == [n for n, _ in model.named_parameters()]
    for (name, a), (_, b) in zip(base.named_parameters(), model.named_parameters()):
        assert torch.equal(a, b), name
    inputs, targets = _batch()
    assert torch.equal(base(inputs, targets), model(inputs, targets, 2))
    assert torch.equal(base(inputs, targets), model.pass_losses(inputs, targets, 3)[0])


@pytest.mark.parametrize("mode", ["glu", "add", "lam", "top"])
def test_trunk_init_and_pass_one_match_the_baseline(mode):
    base = GPT(VOCAB, LAYERS, DIM).float()
    _init(base, 5)
    model = _model(mode, seed=5)
    trunk = dict(base.named_parameters())
    for name, p in model.named_parameters():
        if name in trunk:
            assert torch.equal(p, trunk[name]), name
    inputs, targets = _batch()
    assert torch.equal(base(inputs, targets), model.pass_losses(inputs, targets, 2)[0])


@pytest.mark.parametrize("mode", ["add", "lam", "top"])
def test_zero_initialised_modes_start_as_the_baseline_on_every_pass(mode):
    model = _model(mode)
    inputs, targets = _batch()
    losses = model.pass_losses(inputs, targets, 3)
    base = _model("none").pass_losses(inputs, targets, 1)[0]
    if mode == "top":  # single pass; the zero-initialised read leaves the head input as the plain trunk's
        assert len(losses) == 1 and torch.allclose(losses[0], base) and torch.allclose(model.loss_noread(inputs, targets), base)
        return
    assert torch.allclose(losses[0], losses[1]) and torch.allclose(losses[0], losses[2]) and torch.allclose(losses[0], base)


def test_glu_pass_two_differs_and_first_position_is_plain():
    model = _model("glu")
    _randomize_projections(model)  # the trainer's zero "proj" init makes the untrained trunk the identity
    inputs, targets = _batch()
    losses = model.pass_losses(inputs, targets, 2)
    assert not torch.allclose(losses[0], losses[1])
    state = torch.randn(2, 8, DIM)
    fused = model.fused_input(inputs, state)
    assert torch.allclose(fused[:, 0], model.norm1(model.embed(inputs))[:, 0])
    assert not torch.allclose(fused[:, 1], model.norm1(model.embed(inputs))[:, 1])


def test_linear_memory_parallel_form_matches_the_recurrent_state():
    torch.manual_seed(0)
    B, T, H, d = 2, 6, 2, 4
    q, k, v = torch.randn(B, T, H, d), torch.randn(B, T, H, d), torch.randn(B, T, H, d)
    temperature = torch.tensor([1.0, 0.5])
    decays = torch.tensor(DECAYS)
    qf, kf = linear_memory_features(q, temperature), linear_memory_features(k, temperature)
    parallel = linear_memory_read_parallel(qf, kf, v, decays, chunk=4)
    dense = linear_memory_read_dense(qf, kf, v, decays)
    state = MemoryState(B, H, d, torch.device("cpu"))
    for t in range(T):
        read = state.read(qf[:, t])
        assert torch.allclose(read, parallel[:, t], atol=1e-5), t
        state.write(kf[:, t], v[:, t], decays)
    assert torch.all(parallel[:, 0] == 0)  # nothing to read at the first position
    assert torch.allclose(parallel, dense, atol=1e-5)
    # Longer than one chunk, not a multiple of it, with all heads on different decays: chunked == dense.
    T2, H2 = 37, 4
    q2, k2, v2 = (torch.randn(B, T2, H2, d) for _ in range(3))
    decays2 = torch.tensor((0.5, 0.9, 0.98, 0.999))
    ones = torch.ones(H2)
    qf2, kf2 = linear_memory_features(q2, ones), linear_memory_features(k2, ones)
    chunked = linear_memory_read_parallel(qf2, kf2, v2, decays2, chunk=8)
    assert torch.allclose(chunked, linear_memory_read_dense(qf2, kf2, v2, decays2), atol=1e-4)
    with pytest.raises(ValueError):
        linear_memory_read_parallel(qf2, kf2, v2, decays2, chunk=0)
    assert linear_memory_read_parallel(qf2[:, :0], kf2[:, :0], v2[:, :0], decays2).shape == (B, 0, H2, d)


def test_linear_memory_default_chunk_matches_both_references_over_many_chunks():
    """The production chunk (128) with T > C and the production decays, against the dense form and the state."""
    torch.manual_seed(1)
    B, T, H, d = 1, 260, len(DEFAULT_DECAYS), 8
    q, k, v = (torch.randn(B, T, H, d) for _ in range(3))
    ones = torch.ones(H)
    qf, kf = linear_memory_features(q, ones), linear_memory_features(k, ones)
    decays = torch.tensor(DEFAULT_DECAYS)
    chunked = linear_memory_read_parallel(qf, kf, v, decays)
    assert torch.allclose(chunked, linear_memory_read_dense(qf, kf, v, decays), atol=1e-5)
    state = MemoryState(B, H, d, torch.device("cpu"))
    for t in range(T):
        assert torch.allclose(state.read(qf[:, t]), chunked[:, t], atol=1e-5), t
        state.write(kf[:, t], v[:, t], decays)


def test_memory_reads_and_fusion_are_strictly_causal():
    for mode in ("glu", "add", "lam"):
        model = _model(mode)
        _randomize_projections(model)
        inputs, _ = _batch()
        state = torch.randn(2, 8, DIM)
        bumped = state.clone()
        bumped[:, 5] += 3.0
        a, b = model.pass_forward(inputs, state), model.pass_forward(inputs, bumped)
        assert torch.allclose(a[:, :6], b[:, :6], atol=1e-5), mode  # position 5's own state is invisible to itself
        assert not torch.allclose(a[:, 6:], b[:, 6:]), mode


def _prefix_recompute(model: FeedbackGPT, inputs, targets):
    """The true recurrence via the parallel code: position t uses sequential states of positions < t."""
    B, T = inputs.shape
    states = torch.zeros(B, T, DIM)
    total = torch.zeros(())
    for t in range(T):
        s = model.pass_forward(inputs[:, :t + 1], states[:, :t + 1] if model.mode != "none" else None)
        total = total + model.head_loss(s[:, t], targets[:, t])
        states[:, t] = s[:, t]
    return total


@pytest.mark.parametrize("mode", ["none", "glu", "add", "lam", "top"])
@pytest.mark.parametrize("window", [0, 3])
def test_sequential_evaluator_matches_prefix_recompute(mode, window):
    model = _model(mode, attn_window=window)
    _randomize_projections(model)
    inputs, targets = _batch(B=2, T=8)
    with torch.no_grad():
        expected = _prefix_recompute(model, inputs, targets)
        actual = model.loss_sequential(inputs, targets)
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4), (mode, window, float(actual), float(expected))


@pytest.mark.parametrize(("memory_layers", "window"), [((0,), 0), ((0, 2, 4), 3)])
def test_sparse_lam_is_strictly_causal_and_matches_prefix_recurrence(memory_layers, window):
    model = _model("lam", num_layers=5, memory_layers=memory_layers, attn_window=window)
    _randomize_projections(model)
    inputs, targets = _batch(B=1, T=6)
    state = torch.randn(1, 6, DIM)
    bumped = state.clone()
    bumped[:, 3] += 3.0
    with torch.no_grad():
        original = model.pass_forward(inputs, state)
        changed = model.pass_forward(inputs, bumped)
        assert torch.allclose(original[:, :4], changed[:, :4], atol=1e-5)
        assert not torch.allclose(original[:, 4:], changed[:, 4:])
        assert torch.allclose(model.loss_sequential(inputs, targets), _prefix_recompute(model, inputs, targets),
                              rtol=1e-4, atol=1e-4)


def test_sparse_lam_selection_changes_insertion_site_without_changing_the_plain_trunk():
    first = _model("lam", memory_layers=(0,))
    last = _model("lam", memory_layers=(1,))
    full = _model("lam")
    for model in (first, last, full):
        _randomize_projections(model)
    inputs, _ = _batch()
    state = torch.randn(2, 8, DIM)
    assert sum(p.numel() for p in first.parameters()) < sum(p.numel() for p in full.parameters())
    with torch.no_grad():
        plain = full.pass_forward(inputs, None)
        assert torch.equal(first.pass_forward(inputs, None), plain)
        assert torch.equal(last.pass_forward(inputs, None), plain)
        assert not torch.allclose(first.pass_forward(inputs, state), last.pass_forward(inputs, state))


@pytest.mark.parametrize("memory_layers", [(0,), (0, 2, 4)])
def test_sparse_lam_checkpoint_round_trip_preserves_predictions(memory_layers, tmp_path):
    model = _model("lam", num_layers=5, memory_layers=memory_layers)
    _randomize_projections(model)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_config": model.config, "model": model.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=True)
    rebuilt = FeedbackGPT(**checkpoint["model_config"]).float().eval()
    rebuilt.load_state_dict(checkpoint["model"], strict=True)
    assert rebuilt.config == model.config
    inputs, targets = _batch(B=1, T=4)
    with torch.no_grad():
        assert torch.equal(rebuilt(inputs, targets), model(inputs, targets))
        assert torch.equal(rebuilt.loss_sequential(inputs, targets), model.loss_sequential(inputs, targets))


def test_default_lam_matches_explicit_all_layers_and_loads_legacy_config():
    default = _model("lam")
    explicit = _model("lam", memory_layers=tuple(range(LAYERS)))
    for model in (default, explicit):
        _randomize_projections(model)
    legacy_config = default.config.copy()
    legacy_config.pop("memory_layers")
    restored = FeedbackGPT(**legacy_config).float().eval()
    restored.load_state_dict(default.state_dict(), strict=True)
    inputs, targets = _batch()
    with torch.no_grad():
        assert torch.equal(default(inputs, targets), restored(inputs, targets))
        assert torch.equal(default(inputs, targets), explicit(inputs, targets))
    explicit.load_state_dict(default.state_dict(), strict=True)


@pytest.mark.parametrize("memory_layers", [(), (1, 0), (0, 0), (-1,), (LAYERS,), (0.5,), (True,)])
def test_lam_rejects_invalid_memory_layers(memory_layers):
    with pytest.raises(ValueError):
        _model("lam", memory_layers=memory_layers)


@pytest.mark.parametrize("mode", ["none", "glu", "add", "top"])
def test_other_modes_reject_memory_layer_selection(mode):
    with pytest.raises(ValueError):
        _model(mode, memory_layers=(0,))


def test_window_attention_equals_full_causal_when_it_covers_the_sequence():
    inputs, targets = _batch(B=2, T=8)
    full, wide, narrow = _model("none"), _model("none", attn_window=8), _model("none", attn_window=2)
    for model in (full, wide, narrow):
        _randomize_projections(model)
    assert torch.allclose(full(inputs, targets), wide(inputs, targets), atol=1e-5)
    assert not torch.allclose(full(inputs, targets), narrow(inputs, targets))
    mask = window_mask(4, 2, torch.device("cpu"))
    assert mask.tolist() == [[True, False, False, False], [True, True, False, False],
                             [False, True, True, False], [False, False, True, True]]


def test_detach_stops_the_gradient_through_the_carried_state():
    s = torch.randn(2, 8, DIM, requires_grad=True)
    assert _model("add").carried(s).requires_grad
    assert not _model("add", detach=True).carried(s).requires_grad
    noisy = _model("add", noise=0.1).train()
    torch.manual_seed(0)
    assert not torch.equal(noisy.carried(s), s)
    assert torch.equal(noisy.eval().carried(s), s)


def test_optimizer_partition_covers_every_parameter_once():
    for mode in ("none", "glu", "add", "lam", "top"):
        model = _model(mode)
        extra = model.extra_scalar_parameters()
        extra_ids = {id(p) for p in extra}
        groups = [[model.embed.weight], [model.proj.weight],
                  [p for p in model.parameters() if p.ndim < 2 and id(p) not in extra_ids], extra,
                  [p for p in model.blocks.parameters() if p.ndim >= 2] + model.feedback_matrices()]
        ids = [id(p) for group in groups for p in group]
        assert len(ids) == len(set(ids)) == len(list(model.parameters())), mode


def test_config_round_trips_and_validates():
    model = _model("lam", attn_window=4, detach=True, noise=0.05)
    rebuilt = FeedbackGPT(**model.config)
    assert rebuilt.config == model.config and rebuilt.attn_window == 4
    with pytest.raises(ValueError):
        FeedbackGPT(VOCAB, LAYERS, DIM, mode="gate")
    with pytest.raises(ValueError):
        FeedbackGPT(VOCAB, LAYERS, DIM, mode="lam", memory_head_dim=HEAD, decays=(0.5,))
    with pytest.raises(ValueError):
        FeedbackGPT(VOCAB, LAYERS, DIM, attn_window=-1)


def test_decayed_key_sum_matches_the_dense_normaliser():
    torch.manual_seed(2)
    B, T, H, d = 2, 37, 4, 8
    k = linear_memory_features(torch.randn(B, T, H, d), torch.ones(H))
    decays = torch.tensor(DEFAULT_DECAYS)
    z = decayed_key_sum(k, decays, chunk=8)
    mask = decay_mask(decays, T)  # [H, T, T]
    expected = torch.einsum("hti,bihd->bthd", mask, k)
    assert torch.allclose(z, expected, atol=1e-5)
    assert torch.all(z[:, 0] == 0)


def test_top_mode_reads_only_earlier_top_states_and_detach_cuts_their_gradient():
    inputs, targets = _batch()
    grads = {}
    for detach in (False, True):
        model = _model("top", detach=detach).train()
        _randomize_projections(model)
        # Prefix consistency: position t depends only on tokens <= t (the read is strictly causal).
        with torch.no_grad():
            full = model.pass_forward(inputs, None)
            prefix = model.pass_forward(inputs[:, :5], None)
            assert torch.allclose(full[:, :5], prefix, atol=1e-5)
            # the read changes the head input relative to the plain trunk (mem_proj randomised)
            assert not torch.allclose(model.loss_noread(inputs, targets), model(inputs, targets, 1))
        model(inputs, targets, 1).backward()
        grads[detach] = model.mem_k.weight.grad.clone(), model.blocks[0].mlp.fc.weight.grad.clone()
    # keys/values are used either way (mem_k gets a gradient), but the trunk gradient differs once
    # the path through the earlier top states is cut.
    assert torch.all(grads[True][0] != 0) and torch.all(grads[False][0] != 0)
    assert not torch.allclose(grads[True][1], grads[False][1])

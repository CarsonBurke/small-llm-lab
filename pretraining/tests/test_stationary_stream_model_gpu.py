"""CUDA/BF16 buffer-read, masking, detachment, and ring contracts; execute through mlq."""

import pytest
import torch
import torch.nn.functional as F

from pretraining.future_credit_stream.model import StreamingFFNModel
from pretraining.stationary_stream.model import RingState, StationaryFFNModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

SLOTS, HEADS, KEY_DIM, DIM = 3, 2, 8, 128


def _model(activate=True, entry="value_map", slots=SLOTS):
    torch.manual_seed(31)
    model = StationaryFFNModel(vocab_size=32, model_dim=DIM, num_layers=2, mlp_hidden=256, buffer_slots=slots,
                               read_heads=HEADS, read_key_dim=KEY_DIM, read_entry=entry).cuda()
    if activate:
        # Nonzero head, residuals, and read entry expose every gradient path.
        with torch.no_grad():
            model.proj.weight.normal_(std=0.02)
            model.proj.bias.normal_(std=0.02)
            for block in model.blocks:
                block[1].proj.weight.normal_(std=0.02)
            if entry == "value_map":
                model.read.value.weight.normal_(std=0.05)
                model.read.null_bias.normal_(std=0.5)
            else:
                model.read.query.weight.normal_(std=0.05)
                model.read.age_bias.normal_(std=0.5)
    return model


def _shifted(batch):
    """Random buffer in shift layout: slot j is j+1 steps old."""
    buffer = torch.randn(SLOTS, batch, DIM, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(SLOTS, batch, device="cuda", dtype=torch.bool)
    ages = torch.arange(1, SLOTS + 1, device="cuda")
    return buffer, valid, ages


def _assert_signal(gradient):
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def _rotate(vector, age):
    """Half-truncated RoPE at position ``age`` on the last dim, written out per pair."""
    half = KEY_DIM // 2
    frequencies = [(1 / 1024) ** (i / (KEY_DIM // 4 - 1)) for i in range(KEY_DIM // 4)] + [0.0] * (KEY_DIM // 4)
    result = vector.clone()
    for pair, frequency in enumerate(frequencies):
        angle = torch.tensor(age * frequency, device=vector.device)
        first, second = vector[..., pair], vector[..., half + pair]
        result[..., pair] = first * angle.cos() + second * angle.sin()
        result[..., half + pair] = -first * angle.sin() + second * angle.cos()
    return result


def _reference_read(model, observation, buffer, valid, ages):
    """Independent FP32 equations: rotate keys by age, project every slot's value first, then mix per head."""
    read = model.read
    observation, buffer = observation.float(), buffer.float()
    slots, batch, dim = buffer.shape
    query = F.linear(observation, read.query.weight, read.query.bias).view(batch, HEADS, KEY_DIM)
    keys = F.linear(buffer, read.key.weight, read.key.bias).view(slots, batch, HEADS, KEY_DIM)
    logits = torch.zeros(batch, HEADS, slots + 1, device="cuda")
    logits[:, :, 0] = read.null_bias
    for j in range(slots):
        logits[:, :, j + 1] = (query * _rotate(keys[j], int(ages[j]))).sum(-1) / KEY_DIM ** 0.5
        logits[:, :, j + 1].masked_fill_(~valid[j][:, None], float("-inf"))
    attention = logits.softmax(-1)
    width = dim // HEADS
    result = read.value.bias.expand(batch, dim).clone()
    for h in range(HEADS):
        columns = read.value.weight[:, h * width:(h + 1) * width]
        for j in range(slots):
            projected = F.linear(buffer[j, :, h * width:(h + 1) * width], columns)
            result = result + attention[:, h, j + 1, None] * projected
    return result, attention


def test_read_matches_value_first_reference_and_masks_slots():
    model = _model()
    buffer, valid, ages = _shifted(5)
    valid[1, 0] = False
    valid[:, 2] = False
    valid[0, 4] = False
    valid[2, 4] = False
    observed = torch.tensor([2, 3, 4, 5, 6], device="cuda")
    # Production disables autocast inside forward; the read scores must run in FP32.
    with torch.autocast("cuda", enabled=False):
        observation = model.observation_norm(model.embed(observed).bfloat16())
        read, attention = model.read(observation, buffer, valid, ages)
    expected_read, expected_attention = _reference_read(model, observation, buffer, valid, ages)
    assert read.dtype == torch.bfloat16 and attention.dtype == torch.float32
    assert attention.shape == (5, HEADS, SLOTS + 1)
    torch.testing.assert_close(attention, expected_attention, rtol=2e-2, atol=1e-3)
    torch.testing.assert_close(read.float(), expected_read, rtol=3e-2, atol=2e-2)
    # Masked slots receive exactly no mass; rows still sum to one.
    assert torch.all(attention[0, :, 2] == 0) and torch.all(attention[4, :, 1] == 0) and torch.all(attention[4, :, 3] == 0)
    assert torch.all(attention[2, :, 1:] == 0) and torch.all(attention[2, :, 0] == 1)
    torch.testing.assert_close(attention.sum(-1), torch.ones(5, HEADS, device="cuda"))
    # A lane with no readable slot reads exactly the value bias (zero here).
    assert torch.all(read[2] == 0)
    with torch.no_grad():
        model.read.value.bias.normal_(std=0.1)
    with torch.autocast("cuda", enabled=False):
        biased, _ = model.read(observation, buffer, valid, ages)
    torch.testing.assert_close(biased[2].float(), model.read.value.bias, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("entry", ["value_map", "latent_norm"])
def test_ages_make_the_read_invariant_to_ring_layout(entry):
    """Permuting slots and their ages together changes nothing; changing an age alone does."""
    model = _model(entry=entry)
    buffer, valid, ages = _shifted(4)
    valid[0, 1] = False
    observed = torch.tensor([2, 3, 4, 5], device="cuda")
    order = torch.tensor([2, 0, 1], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _, attention, read = model(observed, buffer, valid, ages)
        permuted_logits, _, permuted_attention, permuted_read = model(
            observed, buffer[order], valid[order], ages[order])
        mislabeled, _, _, _ = model(observed, buffer, valid, ages.flip(0))
    torch.testing.assert_close(permuted_read, read, rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(permuted_logits, logits, rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(permuted_attention[..., 1:], attention[..., 1:][..., order], rtol=1e-3, atol=1e-4)
    assert not torch.allclose(mislabeled, logits, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("entry", ["value_map", "latent_norm"])
def test_reset_lane_is_the_context_free_model_and_untrained_read_is_zero(entry):
    model = _model(entry=entry)
    buffer, valid, ages = _shifted(3)
    observed = torch.tensor([7, 8, 9], device="cuda")
    resets = torch.tensor([False, True, False], device="cuda")
    empty = model.initial_state(3, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, hidden, next_valid = model.step(observed, buffer, valid, ages, resets)
        empty_logits, _, _ = model.step(observed, empty.buffer, empty.valid, empty.ages, resets)
        full_logits, _, _ = model.step(observed, buffer, valid, ages, torch.zeros_like(resets))
    torch.testing.assert_close(logits[1], empty_logits[1], rtol=0, atol=0)
    assert not torch.equal(logits[0], empty_logits[0]) and not torch.equal(logits[2], empty_logits[2])
    assert not torch.equal(logits[1], full_logits[1])
    # The reset lane keeps only the slot the caller is about to commit (age K).
    assert torch.equal(next_valid[:, 1], ages == SLOTS)
    assert torch.all(next_valid[:, 0]) and torch.all(next_valid[:, 2])
    assert hidden.dtype == torch.bfloat16 and not hidden.requires_grad
    if entry == "value_map":
        # An untrained value map is the context-free FFN everywhere.
        fresh = _model(activate=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with_context, _, _, read = fresh(observed, buffer, valid, ages)
            without, _, _, _ = fresh(observed, empty.buffer, empty.valid, empty.ages)
        assert torch.all(read == 0)
        torch.testing.assert_close(with_context, without, rtol=0, atol=0)
    else:
        # An untrained latent_norm entry reads at unit RMS from step 0 with the
        # recency prior alone: attention is softmax(-(age-1)) over readable slots.
        fresh = _model(activate=False, entry="latent_norm")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, attention, read = fresh(observed, buffer, valid, ages)
        prior = (-(ages.float() - 1)).softmax(0)
        torch.testing.assert_close(attention[:, :, 1:], prior.expand(3, HEADS, SLOTS), rtol=1e-5, atol=1e-6)
        assert torch.all(attention[:, :, 0] == 0)
        mixed = (prior[:, None, None].to(buffer.dtype) * buffer).sum(0)
        expected = F.rms_norm(mixed.float(), (DIM,))
        torch.testing.assert_close(fresh.latent_norm(read).float(), expected, rtol=2e-2, atol=1e-2)


def test_ring_commits_into_the_oldest_slot_and_ages_follow():
    model = _model()
    state = model.initial_state(2, "cuda")
    assert state.buffer.shape == (SLOTS, 2, DIM) and state.valid.shape == (SLOTS, 2)
    assert not state.valid.any() and state.buffer.dtype == torch.bfloat16
    # A fresh ring's newest is slot K-1 (age 1), so slot 0 is the oldest and written first.
    assert state.ages.tolist() == list(range(SLOTS, 0, -1)) and state.oldest == 0
    observed = torch.tensor([4, 5], device="cuda")
    resets = torch.tensor([True, False], device="cuda")
    hiddens = []
    for tick in range(SLOTS + 1):
        oldest = state.oldest
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, hidden, next_valid = model.step(observed, state.buffer, state.valid, state.ages,
                                               resets if tick == 0 else torch.zeros_like(resets))
        state.commit(hidden, next_valid)
        hiddens.append((oldest, hidden))
        assert state.newest == oldest
        assert torch.equal(state.buffer[oldest], hidden)
        assert state.ages[oldest] == 1
        # Every readable slot's age is exactly how many commits ago it was written.
        for written_at, (slot, value) in enumerate(hiddens):
            age = len(hiddens) - written_at
            if age <= SLOTS:
                assert state.ages[slot] == age
                assert torch.equal(state.buffer[slot], value)
                assert state.valid[slot].all()
    # After K+1 commits the ring holds the last K hiddens, all valid, newest first by age.
    assert state.valid.all()
    assert sorted(state.ages.tolist()) == list(range(1, SLOTS + 1))
    # Checkpoint round trip.
    saved = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in state.state_dict().items()}
    other = model.initial_state(2, "cuda")
    other.load_state_dict(saved)
    assert torch.equal(other.buffer, state.buffer) and torch.equal(other.valid, state.valid)
    assert other.newest == state.newest
    with pytest.raises(ValueError):
        model.initial_state(3, "cuda").load_state_dict(saved)
    # Reset returns to the empty ring in place: the same tensors, fresh contents.
    tensors = (state.buffer, state.valid, state.ages)
    state.reset()
    assert all(current is original for current, original in zip((state.buffer, state.valid, state.ages), tensors))
    assert not state.valid.any() and not state.buffer.any() and state.oldest == 0
    assert state.ages.tolist() == list(range(SLOTS, 0, -1))


@pytest.mark.parametrize("entry", ["value_map", "latent_norm"])
def test_buffer_is_detached_and_the_read_block_trains_with_the_backbone(entry):
    model = _model(entry=entry)
    buffer, valid, ages = _shifted(3)
    buffer.requires_grad_(True)
    valid[:, 2] = False  # a lane with nothing readable must keep every gradient finite
    observed = torch.tensor([4, 5, 6], device="cuda")
    targets = torch.tensor([7, 8, 9], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _, _, _ = model(observed, buffer, valid, ages)
        F.cross_entropy(logits, targets).backward()
    assert buffer.grad is None
    for name in ("query", "key"):
        _assert_signal(getattr(model.read, name).weight.grad)
    if entry == "value_map":
        _assert_signal(model.read.value.weight.grad)
        _assert_signal(model.read.null_bias.grad)
    else:
        _assert_signal(model.read.age_bias.grad)
        _assert_signal(model.latent_norm.gains.grad)
    _assert_signal(model.embed.weight.grad[observed])
    _assert_signal(model.proj.weight.grad)
    for block in model.blocks:
        _assert_signal(block[1].fc.weight.grad)


def test_lanes_have_independent_outputs_and_read_gradients():
    model = _model()
    buffer, valid, ages = _shifted(3)
    observed = torch.tensor([2, 3, 4], device="cuda")
    resets = torch.zeros(3, device="cuda", dtype=torch.bool)
    changed_observed = observed.clone()
    changed_observed[0] = 8
    changed_buffer = buffer.clone()
    changed_buffer[:, 0] = torch.randn_like(changed_buffer[:, 0])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, hidden, _ = model.step(observed, buffer, valid, ages, resets)
        changed_logits, changed_hidden, _ = model.step(changed_observed, changed_buffer, valid, ages, resets)
    with torch.autocast("cuda", enabled=False):
        observation = model.observation_norm(model.embed(observed).bfloat16()).detach().requires_grad_(True)
        read, _ = model.read(observation, buffer, valid, ages)
        gradient = torch.autograd.grad(read[:1].float().square().sum(), observation)[0]
    for original, changed in ((logits, changed_logits), (hidden, changed_hidden)):
        assert not torch.equal(original[0], changed[0])
        torch.testing.assert_close(original[1:], changed[1:], rtol=0, atol=0)
    _assert_signal(gradient[:1])
    assert torch.count_nonzero(gradient[1:]) == 0


def _reference_latent_norm_read(model, observation, buffer, valid, ages):
    """Independent FP32 equations for the latent_norm entry: recency-biased scores over readable slots,
    no null slot, zero read for a lane with nothing readable, then the control's carry norm."""
    read = model.read
    observation, buffer = observation.float(), buffer.float()
    slots, batch, dim = buffer.shape
    query = F.linear(observation, read.query.weight, read.query.bias).view(batch, HEADS, KEY_DIM)
    keys = F.linear(buffer, read.key.weight, read.key.bias).view(slots, batch, HEADS, KEY_DIM)
    width = dim // HEADS
    mixed = torch.zeros(batch, dim, device="cuda")
    attention = torch.zeros(batch, HEADS, slots + 1, device="cuda")
    for lane in range(batch):
        readable = [j for j in range(slots) if bool(valid[j, lane])]
        if not readable:
            attention[lane, :, 0] = 1
            continue
        for h in range(HEADS):
            scores = torch.stack([(query[lane, h] * _rotate(keys[j, lane, h], int(ages[j]))).sum() / KEY_DIM ** 0.5
                                  + read.age_bias[h, int(ages[j]) - 1] for j in readable])
            weights = scores.softmax(0)
            for weight, j in zip(weights, readable):
                attention[lane, h, j + 1] = weight
                mixed[lane, h * width:(h + 1) * width] += weight * buffer[j, lane, h * width:(h + 1) * width]
    normalized = F.rms_norm(mixed, (dim,), weight=model.latent_norm.gains)
    return normalized, attention


def test_latent_norm_read_matches_reference_and_zeroes_unreadable_lanes():
    model = _model(entry="latent_norm")
    with torch.no_grad():
        model.latent_norm.gains.normal_(mean=1.0, std=0.1)
    buffer, valid, ages = _shifted(5)
    valid[1, 0] = False
    valid[:, 2] = False
    valid[0, 4] = False
    valid[2, 4] = False
    observed = torch.tensor([2, 3, 4, 5, 6], device="cuda")
    with torch.autocast("cuda", enabled=False):
        observation = model.observation_norm(model.embed(observed).bfloat16())
        mixed, attention = model.read(observation, buffer, valid, ages)
        read = model.latent_norm(mixed)
    expected_read, expected_attention = _reference_latent_norm_read(model, observation, buffer, valid, ages)
    assert read.dtype == torch.bfloat16 and attention.dtype == torch.float32
    torch.testing.assert_close(attention, expected_attention, rtol=2e-2, atol=1e-3)
    torch.testing.assert_close(read.float(), expected_read, rtol=3e-2, atol=3e-2)
    assert torch.all(attention[0, :, 2] == 0) and torch.all(attention[4, :, 1] == 0) and torch.all(attention[4, :, 3] == 0)
    assert torch.all(attention[2, :, 1:] == 0) and torch.all(attention[2, :, 0] == 1)
    assert torch.all(read[2] == 0)
    torch.testing.assert_close(attention.sum(-1), torch.ones(5, HEADS, device="cuda"))
    # Every readable lane's read has the norm's RMS profile: unit RMS scaled by the gains.
    for lane in (0, 1, 3, 4):
        expected_rms = model.latent_norm.gains.square().mean().sqrt()
        torch.testing.assert_close(read[lane].float().square().mean().sqrt(), expected_rms, rtol=5e-2, atol=1e-2)


def test_latent_norm_with_one_slot_is_bitwise_the_ce_recursion_control():
    """K=1 latent_norm reads the previous hidden through the control's own carry path."""
    torch.manual_seed(1337)
    control = StreamingFFNModel(vocab_size=32, model_dim=DIM, num_layers=2, mlp_hidden=256, use_writer=False).cuda()
    torch.manual_seed(1337)
    model = _model(activate=False, entry="latent_norm", slots=1)
    with torch.no_grad():  # identical nonzero heads and residuals on both
        model.proj.weight.normal_(std=0.02)
        for block in model.blocks:
            block[1].proj.weight.normal_(std=0.02)
        control.load_state_dict({name: value for name, value in model.state_dict().items()
                                 if not name.startswith("read.")})
    state = model.initial_state(4, "cuda")
    carry = control.initial_state(4, "cuda")
    tokens = torch.tensor([[3, 1, 5, 6], [7, 8, 9, 10], [11, 1, 13, 14], [15, 16, 1, 18], [19, 20, 21, 1]],
                          device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for observed in tokens:
            resets = observed == 1
            logits, hidden, next_valid = model.step(observed, state.buffer, state.valid, state.ages, resets)
            state.commit(hidden, next_valid)
            control_logits, control_hidden = control(observed, carry.masked_fill(resets[:, None], 0.0))
            carry = control_hidden.detach()
            torch.testing.assert_close(logits, control_logits, rtol=0, atol=0)
            torch.testing.assert_close(hidden, control_hidden, rtol=0, atol=0)

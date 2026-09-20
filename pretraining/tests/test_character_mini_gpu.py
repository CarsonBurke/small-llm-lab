"""Causal, normalized character scoring and exact transport. Run through mlq."""

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_character_model import (
    CharacterConfig,
    CharacterGPT,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
VOCAB_SIZE = 37


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(813)
    result = CharacterGPT(CharacterConfig(vocab_size=VOCAB_SIZE)).cuda()
    # Exercise attention and MLP paths, not the identity trunk at initialization.
    with torch.no_grad():
        for name, parameter in result.prior.named_parameters():
            if name.endswith("proj.weight"):
                parameter.normal_(std=0.03)
        result.prior.proj.bias.copy_(
            torch.linspace(-1.5, 1.5, VOCAB_SIZE, device="cuda")
        )
    result.compile_components()
    result.compile(dynamic=False)
    return result


@pytest.mark.parametrize("length", [1, 35])
def test_ce_scores_first_character_and_untrimmed_tail(model, length):
    ids = torch.arange(2 * length, device="cuda").reshape(2, length) % VOCAB_SIZE
    with torch.no_grad():
        logits = model.logits(ids)
        total, stats = model(ids)
        without_auxiliary, same_stats = model(ids, codec_weight=0.0)
    assert logits.shape == (2, length, VOCAB_SIZE)
    # Independent normalized-softmax expression, not an averaged or shifted CE.
    per_character = logits.logsumexp(-1) - logits.gather(-1, ids[..., None]).squeeze(-1)
    torch.testing.assert_close(total, per_character.sum())
    torch.testing.assert_close(stats["rate_nats"], per_character.sum())
    torch.testing.assert_close(without_auxiliary, total, rtol=0, atol=0)
    torch.testing.assert_close(
        same_stats["rate_nats"], stats["rate_nats"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        stats["symbol_accuracy"], (logits.argmax(-1) == ids).float().mean()
    )
    assert set(stats) == {"rate_nats", "symbol_accuracy"}
    assert all(value.ndim == 0 and not value.requires_grad for value in stats.values())
    # All trunk biases are zero: zero-feature BOS must reach the output bias,
    # even though the embedding/trunk/readout weights are nontrivial.
    bias = model.prior.proj.bias.detach().bfloat16().float()
    bos_logits = 15 * bias * (bias.square() + 15**2).rsqrt()
    torch.testing.assert_close(logits[:, 0], bos_logits.expand(2, -1))


def test_single_character_likelihood_is_normalized_over_actual_alphabet(model):
    # For a one-character source, every possible symbol shares the same BOS.
    # Individual forward losses, rather than softmax() itself, define P(symbol).
    losses = []
    with torch.no_grad():
        for symbol in range(VOCAB_SIZE):
            ids = torch.tensor([[symbol]], device="cuda")
            loss, _ = model(ids)
            losses.append(loss)
    probabilities = torch.stack(losses).neg().exp()
    torch.testing.assert_close(probabilities.sum(), probabilities.new_tensor(1.0))
    assert bool((probabilities[1:] > probabilities[:-1]).all())


def test_future_and_current_targets_cannot_change_prefix_predictions(model):
    ids = torch.arange(70, device="cuda").reshape(2, 35) % VOCAB_SIZE
    changed = ids.clone()
    changed[:, 17:] = (changed[:, 17:] + 11) % VOCAB_SIZE
    with torch.no_grad():
        before, after = model.logits(ids), model.logits(changed)
    # Prediction 17 sees source16, never source17; later predictions must see it.
    torch.testing.assert_close(before[:, :18], after[:, :18], rtol=0, atol=0)
    assert bool((before[:, 18:] != after[:, 18:]).any())


def test_tail_is_a_target_not_an_unshifted_embedding_input(model):
    ids = torch.arange(35, device="cuda").unsqueeze(0)
    total, stats = model(ids)
    embedding_gradient, head_gradient = torch.autograd.grad(
        total, (model.table.weight, model.prior.proj.weight)
    )
    assert not stats["rate_nats"].requires_grad
    assert bool(torch.isfinite(embedding_gradient).all())
    assert bool((embedding_gradient[:34] != 0).any())
    assert bool((embedding_gradient[34:] == 0).all())
    assert bool(torch.isfinite(head_gradient).all())
    assert bool((head_gradient != 0).any())


def test_identity_transport_roundtrips_every_symbol_without_neural_calls(
    model, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("identity transport must not execute the neural model")

    monkeypatch.setattr(model, "logits", forbidden)
    monkeypatch.setattr(model.prior.embed, "forward", forbidden)
    monkeypatch.setattr(model.prior.proj, "forward", forbidden)
    ids = torch.arange(VOCAB_SIZE, device="cuda").unsqueeze(0)
    exported = model.export(ids)
    width, residual_width = model.transport_widths
    assert residual_width == 0
    assert exported["residual"] == [[] for _ in range(VOCAB_SIZE)]
    assert exported["latents"] == [
        [(symbol >> shift) & 1 for shift in range(width - 1, -1, -1)]
        for symbol in range(VOCAB_SIZE)
    ]
    assert model.recover(**exported) == list(range(VOCAB_SIZE))
    assert model.export(ids[:, :0]) == {"latents": [], "residual": []}
    assert model.recover([], []) == []


def test_transport_rejects_reserved_ids_and_malformed_streams(model):
    width, _ = model.transport_widths
    for reserved in range(VOCAB_SIZE, 1 << width):
        row = [(reserved >> shift) & 1 for shift in range(width - 1, -1, -1)]
        with pytest.raises(ValueError, match="outside the checkpoint alphabet"):
            model.recover([row], [[]])
    for row in ([0] * (width - 1), [0] * (width + 1), [2] * width, [0.0] * width):
        with pytest.raises(ValueError, match="invalid binary stream or width"):
            model.recover([row], [[]])
    with pytest.raises(ValueError, match="no residual bits"):
        model.recover([[0] * width], [[0]])
    with pytest.raises(ValueError, match="stream lengths differ"):
        model.recover([[0] * width], [])
    with pytest.raises(ValueError, match="stream lengths differ"):
        model.recover([], [[]])
    for invalid in (-1, VOCAB_SIZE):
        with pytest.raises(ValueError, match="outside the checkpoint alphabet"):
            model.export(torch.tensor([[invalid]], device="cuda"))
    with pytest.raises(ValueError, match="must be integers"):
        model.export(torch.tensor([[0.5]], device="cuda"))

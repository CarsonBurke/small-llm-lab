"""Tensor-stub checks for independent, causal post-action binary Q.

These tests establish indexing, masking and gradient wiring. They do not
measure critic calibration, training stability or task improvement.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from postraining.latent_rollout import PAD_SLOT, THOUGHT_SLOT, TOKEN_SLOT
from postraining.value_model import PostActionQHead, SeparateCritic


class TensorTrunk(nn.Module):
    """A deterministic cumulative prefix representation, not a model run."""

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(8, 2)
        with torch.no_grad():
            self.tok_emb.weight.copy_(
                torch.arange(16, dtype=torch.float32).reshape(8, 2) / 8
            )

    def embed_tokens(self, tokens):
        return self.tok_emb(tokens)

    def temporal_belief_from_token_latent(self, inputs):
        return inputs.cumsum(dim=1)


def batch():
    return SimpleNamespace(
        token_ids=torch.tensor([[1, 2, 3, 0], [2, 4, 0, 0]]),
        kind=torch.tensor(
            [[TOKEN_SLOT, TOKEN_SLOT, TOKEN_SLOT, PAD_SLOT],
             [TOKEN_SLOT, TOKEN_SLOT, PAD_SLOT, PAD_SLOT]]
        ),
        action_mask=torch.tensor([[1., 1., 0., 0.], [1., 0., 0., 0.]]),
        thoughts=torch.empty(2, 4, 0),
        hiddens=torch.randn(2, 4, 2, requires_grad=True),
        carry_injected=True,
        memory_carry=True,
    )


def critic(prior=.5):
    result = SeparateCritic(
        TensorTrunk(), post_action_q_hidden=3, post_action_q_prior=prior
    )
    # Positive stub features make the gradient checks deterministic.
    with torch.no_grad():
        result.post_action_q_head.belief.weight.fill_(.25)
    return result


def make_readout_nonzero(q):
    with torch.no_grad():
        q.post_action_q_head.belief.weight.fill_(.25)
        q.post_action_q_head.out.weight.fill_(.4)


def test_logits_use_successor_prefix_including_final_sampled_token():
    q, b = critic(), batch()
    make_readout_nonzero(q)
    inputs = q.trunk.embed_tokens(b.token_ids) * (b.kind != PAD_SLOT)[..., None]
    prefix = inputs.cumsum(1)
    expected = q.post_action_q_head(prefix[:, 1:].float())
    actual = q.action_success_logits(b)
    torch.testing.assert_close(actual[0, :2], expected[0, :2])
    torch.testing.assert_close(actual[1, :1], expected[1, :1])
    # The last sampled action has no following action, but its token remains
    # a valid successor. Its Q includes that token, rather than PAD or t.
    assert actual[0, 1] != q.post_action_q_head(prefix[0, 1])
    assert not actual[~b.action_mask.bool()].any()


def test_future_tokens_do_not_change_earlier_action_q():
    q, b = critic(), batch()
    make_readout_nonzero(q)
    before = q.action_success_logits(b)
    b.token_ids[0, 2] = 7
    after = q.action_success_logits(b)
    torch.testing.assert_close(before[0, 0], after[0, 0], rtol=0, atol=0)
    assert before[0, 1] != after[0, 1]


def test_actor_carry_is_not_an_input_or_gradient_dependency():
    q, b = critic(), batch()
    make_readout_nonzero(q)
    before = q.action_success_logits(b)
    b.hiddens = torch.full_like(b.hiddens, float('nan'), requires_grad=True)
    b.carry_injected = False
    b.memory_carry = False
    after = q.action_success_logits(b)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    after.sum().backward()
    assert b.hiddens.grad is None
    assert q.actor_memory_writer is None
    assert isinstance(q.combiner, nn.Identity)
    assert q.head is None and q.q_head is None


@pytest.mark.parametrize('prior', [.5, .02, .8])
def test_explicit_prior_and_nonaction_probability_mask(prior):
    q, b = critic(prior), batch()
    probabilities, dueling = q.estimates(b)
    expected = b.action_mask * prior
    torch.testing.assert_close(probabilities, expected)
    torch.testing.assert_close(q.values(b), expected)
    assert q.values_actions and dueling is None
    assert probabilities.dtype == torch.float32


def test_bce_gradient_reaches_readout_then_trunk_and_feature_projection():
    q, b = critic(), batch()
    mask = b.action_mask.bool()
    targets = torch.tensor([1., 0., 1.])
    loss = nn.functional.binary_cross_entropy_with_logits(
        q.action_success_logits(b)[mask], targets
    )
    loss.backward()
    assert q.post_action_q_head.out.weight.grad.abs().sum() > 0
    assert q.post_action_q_head.belief.weight.grad.count_nonzero() == 0
    assert q.trunk.tok_emb.weight.grad.count_nonzero() == 0
    q.zero_grad(set_to_none=True)
    make_readout_nonzero(q)
    nn.functional.binary_cross_entropy_with_logits(
        q.action_success_logits(b)[mask], targets
    ).backward()
    assert q.post_action_q_head.belief.weight.grad.abs().sum() > 0
    assert q.trunk.tok_emb.weight.grad.abs().sum() > 0
    assert b.hiddens.grad is None


def test_missing_successor_is_rejected():
    q, b = critic(), batch()
    b.kind[0, 2] = PAD_SLOT
    with pytest.raises(RuntimeError, match='sampled successor'):
        q.action_success_logits(b)


def test_thought_slots_are_rejected_even_without_stored_latents():
    q, b = critic(), batch()
    b.kind[0, 1] = THOUGHT_SLOT
    with pytest.raises(RuntimeError, match='token actions only'):
        q.action_success_logits(b)


@pytest.mark.parametrize('options', [
    {'q_head_hidden': 3},
    {'action_head_hidden': 3},
    {'q_head_reads_actor_belief': True},
    {'actor_memory': object()},
])
def test_other_heads_and_live_actor_writer_are_rejected(options):
    with pytest.raises(ValueError, match='independent token critic'):
        SeparateCritic(TensorTrunk(), post_action_q_hidden=3, **options)


@pytest.mark.parametrize('prior', [0., 1., -.1, float('nan'), float('inf')])
def test_invalid_prior_is_rejected(prior):
    with pytest.raises(ValueError, match='prior'):
        PostActionQHead(2, 3, prior)


def test_default_success_head_prior_is_two_percent():
    head = PostActionQHead(2, 3)
    torch.testing.assert_close(head(torch.zeros(2, 2)).sigmoid(), torch.full((2,), .02))

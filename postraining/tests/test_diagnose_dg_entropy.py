import torch
import torch.nn.functional as F

from postraining.core import delightful_policy_loss
from postraining.diagnose_dg_entropy import (
    delightful_coefficients,
    entropy_proxies,
    sampled_entropy_terms,
)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(-1)


def _proxies_for_every_action(logits: torch.Tensor):
    """Each proxy evaluated as if every vocabulary entry were the sample."""
    vocab = logits.numel()
    terms = sampled_entropy_terms(
        logits.expand(vocab, 1, vocab).repeat(1, 2, 1),
        torch.stack((torch.zeros(vocab, dtype=torch.long), torch.arange(vocab)), 1),
    )
    return entropy_proxies(
        terms["surprisal"][:, 0], terms["entropy"][:, 0], terms["collision"][:, 0]
    )


def test_entropy_proxies_are_exact_expected_first_order_changes():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(7, generator=generator, dtype=torch.float64) * 2
    coefficients = torch.randn(7, generator=generator, dtype=torch.float64)
    probs = logits.softmax(-1)
    proxies = _proxies_for_every_action(logits.float())

    entropy_grad = torch.func.grad(_entropy)(logits)
    # Natural-gradient step on the logits: delta z_a = c_a.
    natural = entropy_grad @ coefficients
    # Plain expected score step: delta z = E_a[c_a (e_a - pi)].
    score_step = probs * coefficients - probs * (probs @ coefficients)
    softmax = entropy_grad @ score_step

    torch.testing.assert_close(
        probs @ (coefficients * proxies["natural"].double()), natural,
        rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        probs @ (coefficients * proxies["softmax"].double()), softmax,
        rtol=1e-5, atol=1e-6,
    )


def test_sampled_terms_are_indexed_by_the_acting_slot():
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 5, 11, generator=generator)
    token_ids = torch.randint(0, 11, (2, 5), generator=generator)
    terms = sampled_entropy_terms(logits, token_ids)
    log_probs = logits.log_softmax(-1)
    # Slot 2's action emits token 3 from slot 2's logits.
    torch.testing.assert_close(
        terms["surprisal"][:, 2], -log_probs[torch.arange(2), 2, token_ids[:, 3]]
    )
    torch.testing.assert_close(terms["entropy"][:, 2], _entropy(logits[:, 2]))
    for value in terms.values():
        assert value.shape == token_ids.shape
        assert torch.all(value[:, -1] == 0)


def test_delightful_coefficients_match_the_training_loss_gradient():
    generator = torch.Generator().manual_seed(2)
    logprobs = -torch.rand(3, 6, generator=generator) * 4
    advantages = torch.randn(3, 6, generator=generator) * 0.5
    mask = torch.ones(3, 6)
    logprobs.requires_grad_(True)
    loss, _ = delightful_policy_loss(logprobs, advantages, mask, denominator=torch.tensor(1.0))
    (grad,) = torch.autograd.grad(loss, logprobs)
    torch.testing.assert_close(
        -grad, delightful_coefficients(advantages, -logprobs.detach(), 1.0)
    )

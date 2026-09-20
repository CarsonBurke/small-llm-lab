"""Output-head split-backward contracts; execute only through mlq.

The fixture isolates the real D512/V1024 softcapped head and chunked sum-CE.
It deliberately omits recurrent computation: this is an execution-equivalence
check, not model-quality or recurrent-training evidence.
"""
import os

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from pretraining.nanogpt_mini.nanogpt_mini_model import Linear
from pretraining.nanogpt_mini.recurrent_slots_runtime import RecurrentLoss, CUDAGraphMicrobatch
from pretraining.nanogpt_mini.latent_carry_runtime import LatentCarryLoss


@pytest.fixture(autouse=True)
def cuda_only():
    if not torch.cuda.is_available():
        if os.environ.get("RECURRENT_SLOTS_REQUIRE_CUDA") == "1":
            pytest.fail("queued head split contracts require CUDA")
        pytest.skip("CUDA contracts require mlq")
    assert torch.cuda.is_bf16_supported(), "BF16 required"


class HeadFixture(nn.Module):
    """A leaf hidden tensor exposes every upstream activation gradient."""

    def __init__(self, rows):
        super().__init__()
        self.config = dict(vocab_size=1024)
        # Production recurrent hidden activations are FP32; the dense head
        # still executes in BF16 under the caller's autocast context.
        self.hidden = nn.Parameter(torch.randn(1, rows, 512, device="cuda", dtype=torch.float32))
        self.proj = Linear(512, 1024).cuda()
        with torch.no_grad():
            self.proj.weight.normal_(std=.02)
            self.proj.bias.normal_(std=.01)
        # There is no segment computation to compile in this head-only fixture.
        self._compiled_segment = True

    def forward_hidden(self, inputs, segment_size):
        # Token-dependent multiplication makes both loss and hidden gradients
        # sensitive to changed captured inputs, without another model layer.
        factor = 1 + inputs.to(torch.bfloat16).unsqueeze(-1) / 1024
        return self.hidden * factor, None

    def logits(self, hidden):
        logits = self.proj(hidden).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()


class SliceReferenceLoss(RecurrentLoss):
    """The previous independent-slice head loss, retained only as an oracle."""

    def __call__(self, inputs, targets):
        hidden, _ = self.model.forward_hidden(inputs, segment_size=self.segment_size)
        hidden, targets = hidden.flatten(0, 1), targets.flatten()
        losses = []
        for start in range(0, hidden.shape[0], 4096):
            losses.append(checkpoint(self.head_loss, hidden[start:start + 4096],
                                     targets[start:start + 4096], use_reentrant=False,
                                     preserve_rng_state=False))
        return torch.stack(losses).sum()


def gradient_snapshot(model):
    result = {}
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert float(p.grad.float().norm()) > 0, name
        result[name] = p.grad.detach().clone()
    return result


def assert_gradients(model, expected):
    assert set(expected) == {"hidden", "proj.weight", "proj.bias"}
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        difference = (p.grad.float() - expected[name].float()).norm()
        scale = expected[name].float().norm().clamp_min(1e-12)
        # Identical head shapes and BF16 arithmetic: substantially tighter
        # than architecture/kernel parity tolerances used elsewhere.
        assert float(difference / scale) < 1e-4, name
    # Compare all hidden entries, including both boundaries and the short tail,
    # so a missing/scrambled chunk cannot hide in a projection-gradient sum.
    torch.testing.assert_close(model.hidden.grad, expected["hidden"], rtol=1e-3, atol=1e-7)


@pytest.mark.parametrize("loss_class", [RecurrentLoss, LatentCarryLoss], ids=["recurrent", "latent_carry"])
def test_split_head_matches_slice_gradients_and_captured_updates(loss_class):
    torch._dynamo.reset()
    torch.manual_seed(771)
    rows = 2 * 4096 + 3
    reference_model, actual_model = HeadFixture(rows), HeadFixture(rows)
    actual_model.load_state_dict(reference_model.state_dict(), strict=True)
    reference = SliceReferenceLoss(reference_model, segment_size=1)
    actual = loss_class(actual_model, segment_size=1)
    inputs = torch.randint(1024, (1, rows), device="cuda", dtype=torch.int32)
    targets = torch.randint(1024, (1, rows), device="cuda", dtype=torch.int64)

    # Ordinary compiled head execution first checks the changed autograd
    # structure independently of CUDA graph capture.
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        reference_loss = reference(inputs, targets)
        actual_loss = actual(inputs, targets)
    reference_loss.backward()
    actual_loss.backward()
    expected = gradient_snapshot(reference_model)
    torch.testing.assert_close(actual_loss.detach(), reference_loss.detach(), rtol=1e-6, atol=1e-3)
    assert_gradients(actual_model, expected)
    del reference_loss, actual_loss

    graph = CUDAGraphMicrobatch(actual, batch_size=1, seq_len=rows)
    pointers = {name: p.grad.data_ptr() for name, p in actual_model.named_parameters()}
    first_expected = expected
    for update in range(2):
        if update:
            inputs = (inputs + 317).remainder(1024)
            targets = (targets + 97).remainder(1024)
            with torch.no_grad():
                reference_model.proj.weight.mul_(.93)
                reference_model.proj.bias.add_(.02)
                reference_model.hidden.mul_(1.03)
                actual_model.load_state_dict(reference_model.state_dict(), strict=True)
        reference_model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            reference_loss = reference(inputs, targets)
            reference_loss.backward()
        expected = gradient_snapshot(reference_model)
        if update:
            relative_change = ((expected["hidden"].float() - first_expected["hidden"].float()).norm()
                               / first_expected["hidden"].float().norm())
            assert float(relative_change) > .05, "mutation check is insensitive"
        graph.zero_grad()
        captured_loss = graph.replay(inputs, targets).clone()
        torch.testing.assert_close(captured_loss, reference_loss.detach(), rtol=1e-6, atol=1e-3)
        assert_gradients(actual_model, expected)
        for name, p in actual_model.named_parameters():
            assert p.grad.data_ptr() == pointers[name], name
        del reference_loss

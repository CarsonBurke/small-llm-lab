"""Causal Mini priors with normalized mixture or prefix-conditioned bit densities."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pretraining.nanogpt_mini.nanogpt_mini_model import Block, Linear, RMSNorm


def binary_st(logits: Tensor) -> Tensor:
    """Hard binary forward, derivative of the Bernoulli probability relaxation."""
    soft = logits.sigmoid()
    return (logits > 0).to(soft.dtype) + (soft - soft.detach())


def binary_nll(parameters: Tensor, targets: Tensor, components: int) -> Tensor:
    """Per-position nats over the full binary domain; mixture IDs are marginalized."""
    if components == 1:
        return F.binary_cross_entropy_with_logits(
            parameters, targets, reduction="none"
        ).sum(-1)
    joint = parameters.unflatten(-1, (components, targets.shape[-1] + 1))
    bit_logits = joint[..., 1:]
    # BCE's fused operator requires equal shapes, not implicit broadcasting.
    component_nll = F.binary_cross_entropy_with_logits(
        bit_logits, targets.unsqueeze(-2).expand_as(bit_logits), reduction="none"
    ).sum(-1)
    return -torch.logsumexp(
        F.log_softmax(joint[..., 0], dim=-1) - component_nll, dim=-1
    )


def bit_entropy(bits: Tensor) -> Tensor:
    """Mean marginal bit entropy; balanced constant vectors still have zero entropy."""
    probability = bits.detach().mean((0, 1))
    return -(
        torch.xlogy(probability, probability)
        + torch.xlogy(1 - probability, 1 - probability)
    ).mean() / math.log(2)


class PrefixBinaryHead(nn.Module):
    """Shared bit decoder; each decision sees context and strictly earlier bits."""

    def __init__(self, model_dim: int, bits: int, width: int):
        super().__init__()
        self.context = Linear(model_dim, width)
        self.prefix = nn.Linear(bits - 1, width, bias=False)
        self.position = nn.Embedding(bits, width)
        self.output = Linear(width, 1)

    def forward(self, context: Tensor, targets: Tensor) -> Tensor:
        signed = (2 * targets[..., :-1] - 1).to(torch.bfloat16)
        contributions = signed.unsqueeze(-1) * self.prefix.weight.T.to(torch.bfloat16)
        # Exclusive prefix sum: decision j cannot see its own bit or its suffix.
        prefix = F.pad(contributions.cumsum(dim=-2), (0, 0, 1, 0))
        hidden = (
            self.context(context.to(torch.bfloat16)).unsqueeze(-2)
            + prefix
            + self.position.weight.to(torch.bfloat16)
        )
        return self.output(hidden.relu().square()).squeeze(-1)


@torch.compile(dynamic=True, fullgraph=True)
def sample_prefix_code_and_logits(
    head: PrefixBinaryHead,
    context: Tensor,
    uniforms: Tensor,
    temperature: float,
    prefix_code: Tensor | None = None,
    prefix_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample the full code domain, optionally preserving an accepted bit prefix."""
    if (prefix_code is None) != (prefix_mask is None):
        raise ValueError("prefix code and mask must be supplied together")
    bits = head.position.num_embeddings
    code = torch.zeros(
        (context.shape[0], bits), dtype=torch.float32, device=context.device
    )
    projected = head.context(context.to(torch.bfloat16))
    logits = []
    for bit in range(bits):
        # Match PrefixBinaryHead's BF16 cumsum; a repeatedly rounded running
        # sum would change its autoregressive conditionals.
        signed = (2 * code[..., :-1] - 1).to(torch.bfloat16)
        contributions = signed.unsqueeze(-1) * head.prefix.weight.T.to(torch.bfloat16)
        prefix = F.pad(contributions.cumsum(dim=-2), (0, 0, 1, 0))
        hidden = (
            projected + prefix[:, bit, :] + head.position.weight[bit].to(torch.bfloat16)
        )
        logit = head.output(hidden.relu().square()).squeeze(-1).float()
        logit = 15 * logit * (logit.square() + 225).rsqrt()
        logits.append(logit)
        if temperature == 0:
            decision = logit > 0
        else:
            decision = uniforms[:, bit] < torch.sigmoid(logit / temperature)
        if prefix_mask is not None:
            decision = torch.where(
                prefix_mask[:, bit], prefix_code[:, bit].bool(), decision
            )
        code[:, bit] = decision.to(torch.float32)
    return code, torch.stack(logits, dim=-1)


@torch.compile(dynamic=True, fullgraph=True)
def sample_prefix_code(
    head: PrefixBinaryHead, context: Tensor, uniforms: Tensor, temperature: float
) -> Tensor:
    """Sample all binary addresses, including reserved ones, without renormalizing."""
    return sample_prefix_code_and_logits(head, context, uniforms, temperature)[0]


class LatentPrior(nn.Module):
    """Unchanged Mini trunk, one binary vector per causal Transformer position."""

    def __init__(self, config):
        super().__init__()
        self.bits = config.latent_bits
        self.components = config.mixture_components
        self.density_head = getattr(config, "density_head", "mixture")
        self.input = Linear(config.latent_bits, config.model_dim)
        self.blocks = nn.ModuleList(
            Block(config.model_dim) for _ in range(config.num_layers)
        )
        self.norm1 = RMSNorm(config.model_dim)
        self.norm2 = RMSNorm(config.model_dim)
        self.proj = Linear(config.model_dim, config.latent_bits)
        if self.density_head == "prefix":
            # Keep codec/input/trunk initialization matched to the mixture control.
            with torch.random.fork_rng(devices=[]):
                self.proj = PrefixBinaryHead(
                    config.model_dim, self.bits, config.prefix_width
                )
        elif self.components > 1:
            # Preserve the independent-head constructor's RNG consumption so
            # changing only M does not also change the native codec initialization.
            with torch.random.fork_rng(devices=[]):
                self.proj = Linear(config.model_dim, self.components * (self.bits + 1))

    @torch.no_grad()
    def reset_mixture_head(self):
        if self.components == 1:
            return
        bias = self.proj.bias.view(self.components, self.bits + 1)
        bias[:, 0].zero_()
        # Identical zero-initialized components cannot specialize. Deterministic
        # small distinct biases break that symmetry without changing RNG state.
        index = torch.arange(
            self.components * self.bits, device=bias.device, dtype=torch.float32
        ).reshape(self.components, self.bits)
        bias[:, 1:].copy_(0.1 * torch.sin(index + 1))

    def forward(self, latents):
        features = 2 * latents - 1
        previous = F.pad(features[:, :-1], (0, 0, 1, 0))
        h = self.norm1(self.input(previous.to(torch.bfloat16)))
        for block in self.blocks:
            h = block(h)
        h = self.norm2(h)
        logits = (
            self.proj(h, latents) if self.density_head == "prefix" else self.proj(h)
        ).float()
        return 15 * logits * (logits.square() + 225).rsqrt()

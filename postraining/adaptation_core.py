"""Pure batch-construction and loss math for latent-thought adaptation.

Adaptation is teacher-forced pretraining plus thought exposure: at a sampled
subset of positions an imagined next-token latent is inserted into the stream
between token t and token t+1.  Every stream position keeps predicting the
next real token — a thought position targets the same upcoming token as the
token before it, so extra latent compute is trained to sharpen an imminent
prediction rather than being loss-masked filler.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

TOKEN_KIND, THOUGHT_KIND, PAD_KIND = 0, 1, -1


@dataclass
class ThoughtPlan:
    """Row-aligned layout of a token stream with thoughts interleaved.

    All tensors are (batch, stream_length).  ``source`` holds, per stream
    position, the token position whose latent feeds that slot: a TOKEN slot
    consumes the token's own latent, a THOUGHT slot consumes a thought sampled
    from the belief after that same token.  ``target`` indexes the token
    position whose successor is predicted at that slot (equal to ``source``
    for both kinds), and PAD slots carry target 0 with zero weights.
    """

    kind: Tensor
    source: Tensor
    target: Tensor
    ce_weight: Tensor
    latent_weight: Tensor

    @property
    def stream_length(self) -> int:
        return self.kind.size(1)


def build_thought_plan(
    insert_mask: Tensor,
    thought_ce_weight: float = 1.0,
    thought_latent_weight: float = 1.0,
) -> ThoughtPlan:
    """Interleave thoughts after masked token positions.

    ``insert_mask`` is a boolean (batch, length) tensor; True at position t
    inserts one thought between token t and token t+1.  Thought slots carry
    separately tunable CE and latent-loss weights: beliefs after a thought
    never occur in pretraining's closed loop, so both deviations are knobs
    rather than baked in.
    """
    if insert_mask.dtype != torch.bool:
        raise ValueError("insert_mask must be boolean")
    batch, length = insert_mask.shape
    device = insert_mask.device
    inserted = insert_mask.long().cumsum(dim=1)
    stream_length = length + int(inserted[:, -1].max()) if length else 0
    kind = torch.full((batch, stream_length), PAD_KIND, dtype=torch.long, device=device)
    source = torch.zeros((batch, stream_length), dtype=torch.long, device=device)

    positions = torch.arange(length, device=device).expand(batch, length)
    # Token t lands after every thought inserted before it; its own thought
    # (if any) lands immediately after it.
    token_slots = positions + inserted - insert_mask.long()
    kind.scatter_(1, token_slots, TOKEN_KIND)
    source.scatter_(1, token_slots, positions)
    thought_slots = (positions + inserted)[insert_mask]
    thought_rows = torch.nonzero(insert_mask, as_tuple=True)[0]
    kind[thought_rows, thought_slots] = THOUGHT_KIND
    source[thought_rows, thought_slots] = positions[insert_mask]

    ce_weight = torch.zeros((batch, stream_length), dtype=torch.float32, device=device)
    ce_weight[kind == TOKEN_KIND] = 1.0
    ce_weight[kind == THOUGHT_KIND] = thought_ce_weight
    latent_weight = torch.zeros((batch, stream_length), dtype=torch.float32, device=device)
    latent_weight[kind == TOKEN_KIND] = 1.0
    latent_weight[kind == THOUGHT_KIND] = thought_latent_weight
    return ThoughtPlan(
        kind=kind,
        source=source,
        target=source.clone(),
        ce_weight=ce_weight,
        latent_weight=latent_weight,
    )


def assemble_stream_inputs(
    plan: ThoughtPlan, token_latents: Tensor, thought_inputs: Tensor
) -> Tensor:
    """Gather (batch, stream, dim) inputs from token latents and thoughts.

    ``thought_inputs`` is (batch, length, dim): the injected thought for the
    slot after each token position (only masked positions are ever read).
    PAD slots receive zeros; causal attention keeps them from influencing any
    earlier position, and their losses carry zero weight.
    """
    gathered_tokens = token_latents.gather(
        1, plan.source[..., None].expand(-1, -1, token_latents.size(-1))
    )
    gathered_thoughts = thought_inputs.gather(
        1, plan.source[..., None].expand(-1, -1, thought_inputs.size(-1))
    )
    inputs = torch.where(
        (plan.kind == THOUGHT_KIND)[..., None], gathered_thoughts, gathered_tokens
    )
    return inputs * (plan.kind != PAD_KIND)[..., None].to(inputs.dtype)


def gather_stream_targets(plan: ThoughtPlan, per_token_targets: Tensor) -> Tensor:
    """Map (batch, length, ...) per-token-position targets onto stream slots."""
    if per_token_targets.dim() == 2:
        return per_token_targets.gather(1, plan.target)
    return per_token_targets.gather(
        1, plan.target[..., None].expand(-1, -1, per_token_targets.size(-1))
    )


def weighted_cross_entropy(
    logits: Tensor, targets: Tensor, weights: Tensor
) -> Tensor:
    """Mean CE over weighted stream slots (natural log, fp32)."""
    flat = torch.nn.functional.cross_entropy(
        logits.float().flatten(0, 1), targets.flatten(), reduction="none"
    )
    weights = weights.flatten()
    return (flat * weights).sum() / weights.sum().clamp_min(1.0)


def copy_last_latent_mse(token_latents: Tensor, target_latents: Tensor) -> Tensor:
    """Baseline the transition must beat: predict no change in the latent."""
    return torch.nn.functional.mse_loss(
        token_latents.float(), target_latents.float()
    )

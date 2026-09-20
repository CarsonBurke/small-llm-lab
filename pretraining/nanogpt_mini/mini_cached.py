"""One-position Mini KV arithmetic shared by genuine character/event runtimes.

Call from compiled kernels with BF16 projected features. This is the actual Mini
block computation, not an adapter between different model architectures.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def rotary_at(x: Tensor, angular_freq: Tensor, position: int) -> Tensor:
    pos = torch.scalar_tensor(position, dtype=torch.float32, device=x.device)
    theta = pos * angular_freq[None, None, None, :]
    cosine, sine = theta.cos(), theta.sin()
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat(
        (first * cosine + second * sine, first * (-sine) + second * cosine), dim=-1
    ).type_as(x)


def cached_mini_state(stack, features: Tensor, keys, values, position: int) -> Tensor:
    """Append one position, using the caller's RAW or EVENT position semantics."""
    x = stack.norm1(features.unsqueeze(1))
    end = position + 1
    for block, key_cache, value_cache in zip(stack.blocks, keys, values):
        attention = block.attn
        normalized = block.norm1(x)
        shape = (1, 1, attention.num_heads, attention.head_dim)
        q = F.rms_norm(attention.q(normalized).view(shape), (attention.head_dim,))
        k = F.rms_norm(attention.k(normalized).view(shape), (attention.head_dim,))
        v = attention.v(normalized).view(shape)
        q = rotary_at(q, attention.rotary.angular_freq, position)
        k = rotary_at(k, attention.rotary.angular_freq, position)
        key_cache[:, :, position:end].copy_(k.transpose(1, 2))
        value_cache[:, :, position:end].copy_(v.transpose(1, 2))
        # Single final-position query: all initialized prefix keys are causal.
        # Rectangular is_causal=True incorrectly hides cached history here.
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            key_cache[:, :, :end],
            value_cache[:, :, :end],
            scale=0.12,
            is_causal=False,
        ).transpose(1, 2)
        x = x + attention.proj(attended.contiguous().view(1, 1, -1))
        x = x + block.mlp(block.norm2(x))
    return stack.norm2(x).squeeze(1)


def allocate_cache(stack, capacity: int, device: torch.device):
    keys, values = [], []
    for block in stack.blocks:
        shape = (1, block.attn.num_heads, capacity, block.attn.head_dim)
        keys.append(torch.empty(shape, dtype=torch.bfloat16, device=device))
        values.append(torch.empty(shape, dtype=torch.bfloat16, device=device))
    return tuple(keys), tuple(values)


@torch.compile(dynamic=True, fullgraph=True)
def sample_character(logits: Tensor, uniform: Tensor, temperature: float) -> Tensor:
    if temperature == 0:
        return logits.argmax(-1)
    cumulative = (logits / temperature).softmax(-1).cumsum(-1)
    return (cumulative < uniform).sum(-1).clamp_max(logits.shape[-1] - 1)

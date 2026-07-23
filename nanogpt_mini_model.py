"""nanogpt_mini_model.py

Model classes shared by the nanogpt-mini training scripts
(``nanogpt_mini_train.py``, ``nanogpt_mini_tieddot_train.py``) and by the
post-training stack (``postraining/nano_backbone.py``), extracted verbatim so
the architecture is importable without executing a training run (the scripts
run torchrun setup and the training loop at module scope).

Extraction contract:
  - Module code is byte-identical to the definitions previously inlined in the
    scripts, with two mechanical parameterizations that preserve behavior:
      1. ``MLP``/``Block``/``GPT`` accept an optional ``mlp_hidden`` width;
         ``None`` means the baseline's ``4 * dim``. The tieddot script passes
         its ``MLP_HDIM`` env value through instead of reading the env inside
         ``MLP.__init__``.
      2. The tieddot head lives here as ``TiedDotGPT`` (the scripts each
         called their own variant ``GPT``).
  - Submodule registration order is unchanged in every class, so the scripts'
    seeded ``named_parameters`` init loop draws identical RNG and short seeded
    runs reproduce pre-extraction val lines bit-for-bit.
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: "int | None" = None):
        super().__init__()
        hdim = 4 * dim if hidden_dim is None else hidden_dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, mlp_hidden: "int | None" = None):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim, mlp_hidden)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 mlp_hidden: "int | None" = None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, mlp_hidden) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

class TiedEnergyGPT(nn.Module):
    """Machinery-free energy readout: squared-distance scoring against the
    RAW (unnormalized) attached embedding table. Distinct from TiedDotGPT
    only in the score form — with an rms-normed codebook, distance collapses
    to the dot product up to per-row constants, so the unnormalized table is
    what makes this a separate cell (per-row norms act as a learned prior).
    No projectors/BatchNorm/per-dim scales (the lejepa-family machinery),
    just the scalar scale + per-token bias the dot head also has. CAPLESS:
    distance logits are one-sided, so the symmetric softcap would saturate
    the target's logit whenever its code is far; capless CE is also
    shift-invariant per position, making the ||z||^2 term harmless.
    Scoring runs in fp32: energies are O(dim), and bf16's ~2^-8 relative
    quantum at that magnitude would destroy cross-token differences."""
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 mlp_hidden: "int | None" = None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, mlp_hidden) for _ in range(num_layers)])
        # registered after the norms so the shared trunk consumes RNG in
        # baseline order (proj was zero-init and drew none; these draw none)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.readout_scale = nn.Parameter(torch.zeros(()))
        self.readout_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        z = self.norm2(x).float()
        codebook = self.embed.weight.float()  # attached, raw: grads flow, norms free
        sq_dist = (z * z).sum(-1, keepdim=True) - 2.0 * (z @ codebook.t()) \
            + (codebook * codebook).sum(-1)
        logits = self.readout_scale * (-0.5 * sq_dist) + self.readout_bias
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

class TiedDotGPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 mlp_hidden: "int | None" = None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, mlp_hidden) for _ in range(num_layers)])
        # tied attached dot readout replaces the untied proj head; registered
        # after the norms so the shared trunk consumes RNG in baseline order
        # (proj was zero-init and drew none; these draw none either)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.readout_scale = nn.Parameter(torch.zeros(()))
        self.readout_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        z = self.norm2(x)
        # attached codebook: gradients flow into embed through the readout
        codebook = F.rms_norm(self.embed.weight, (self.embed.embedding_dim,))
        logits = (z @ codebook.type_as(z).t()).float() * self.readout_scale + self.readout_bias
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

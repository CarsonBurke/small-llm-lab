"""nanogpt_mini_gpt2vocab_kda_3to1_pm_train.py

Kimi Delta Attention ablation on ``nanogpt_mini_gpt2vocab_train.py``.

The default architecture preserves six complete dense nanoGPT blocks and
inserts eighteen attention-only KDA residual mixers in the repeating
Kimi-style 3:1 hybrid schedule:

    KDA, KDA, KDA, dense  (repeated six times)

``DELTA_LAYER_INDICES`` accepts a comma-separated zero-based layer schedule
and ``NUM_LAYERS`` controls total depth for explicit ablations. KDA insertions
omit their own MLP by default so the schedule retains six rather than
twenty-four expensive MLPs. ``DELTA_MLP_ON_DELTA=1`` restores full KDA blocks
for a direct architectural ablation.

KDA follows the released Kimi-K3 wrapper: three 128-wide q/k/v heads,
causal SiLU short convolutions of width four, channel-wise bounded
decay, scalar beta per head, head-wise gated RMSNorm, a low-rank sigmoid
output gate, and the official FLA chunk kernel. Dense layers retain the
baseline RoPE SDPA attention so this probes KDA rather than simultaneously
changing the global-attention design to MLA.

KDA uses a 384-wide projection and the released Kimi low-rank output gate.
Dense blocks and optional full delta blocks use the baseline MLP width 2048.
``DELTA_ATTENTION_TYPE=gdn2`` substitutes full Gated DeltaNet 2 at the same
head/state sizes, deliberately allowing its additional parameters. GDN-2
uses the NVIDIA reference's independent
channel-wise sigmoid erase/write gates, unbounded channel-wise decay,
log-uniform decay-rate initialization, and SiLU-gated RMSNorm output.
``DELTA_ATTENTION_TYPE=gdn2_kda_erase`` instead starts on the exact KDA
subspace and adds only a zero-initialized low-rank channel-wise erase
residual. This preserves KDA's scalar erase/write prior while allowing the
most useful GDN-2 degree of freedom to emerge during training.
``DELTA_DISABLE_RECOMPUTE=1`` retains kernel intermediates to accelerate
either backward pass.

Requires ``fla-core==0.5.2``. This remains a diagnostic-only GPT-2-vocab
model and is not a 16 MB submission candidate.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import itertools
import json
import math
import resource
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist

try:
    import fla
    from fla.modules import FusedRMSNormGated, FusedRMSNormSwishGate, ShortConvolution
    from fla.ops.gdn2 import chunk_gdn2
    from fla.ops.kda import chunk_kda
except ImportError as exc:
    raise RuntimeError(
        "This KDA ablation requires fla-core==0.5.2. "
        "Install it in an isolated environment before launching."
    ) from exc

VOCAB_SIZE = 50304  # GPT-2 BPE (50,257 real tokens, padded for GPU efficiency)
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "24"))
if NUM_LAYERS <= 0:
    raise ValueError(f"NUM_LAYERS must be positive, got {NUM_LAYERS}")
default_delta_layer_indices = ",".join(
    str(layer_idx)
    for layer_idx in range(NUM_LAYERS)
    if (layer_idx + 1) % 4
)
DELTA_LAYER_INDICES = frozenset(
    int(index)
    for index in os.environ.get(
        "DELTA_LAYER_INDICES",
        default_delta_layer_indices,
    ).split(",")
    if index
)
if not DELTA_LAYER_INDICES.issubset(range(NUM_LAYERS)):
    raise ValueError(
        f"DELTA_LAYER_INDICES must be within [0, {NUM_LAYERS}), "
        f"got {sorted(DELTA_LAYER_INDICES)}"
    )
KDA_NUM_HEADS = 3
DELTA_ATTENTION_TYPE = os.environ.get("DELTA_ATTENTION_TYPE", "kda")
if DELTA_ATTENTION_TYPE not in {"kda", "gdn2", "gdn2_kda_erase"}:
    raise ValueError(
        "DELTA_ATTENTION_TYPE must be 'kda', 'gdn2', or "
        "'gdn2_kda_erase', "
        f"got {DELTA_ATTENTION_TYPE!r}"
    )
DELTA_DISABLE_RECOMPUTE = (
    os.environ.get(
        "DELTA_DISABLE_RECOMPUTE",
        os.environ.get("KDA_DISABLE_RECOMPUTE", "0"),
    )
    == "1"
)
DELTA_STATE_V_FIRST = os.environ.get("DELTA_STATE_V_FIRST", "1") == "1"
KDA_COMPILE_DIAGNOSTICS = os.environ.get("KDA_COMPILE_DIAGNOSTICS", "0") == "1"
DELTA_EAGER_MODULE = os.environ.get("DELTA_EAGER_MODULE", "0") == "1"
DELTA_BLOCKWISE_COMPILE = (
    os.environ.get("DELTA_BLOCKWISE_COMPILE", "1") == "1"
)
DELTA_COMPILE_MODE = os.environ.get("DELTA_COMPILE_MODE", "default")
if DELTA_COMPILE_MODE not in {
    "default",
    "reduce-overhead",
    "max-autotune-no-cudagraphs",
    "max-autotune",
}:
    raise ValueError(
        "DELTA_COMPILE_MODE must be 'default', 'reduce-overhead', "
        "'max-autotune-no-cudagraphs', or 'max-autotune', "
        f"got {DELTA_COMPILE_MODE!r}"
    )
DELTA_USE_CUDAGRAPHS = DELTA_COMPILE_MODE in {
    "reduce-overhead",
    "max-autotune",
}
DELTA_MLP_ON_DELTA = os.environ.get("DELTA_MLP_ON_DELTA", "0") == "1"
MLP_HIDDEN = int(os.environ.get("MLP_HIDDEN", "2048"))
GDN2_RESIDUAL_RANK = int(os.environ.get("GDN2_RESIDUAL_RANK", "32"))
if MLP_HIDDEN <= 0:
    raise ValueError(f"MLP_HIDDEN must be positive, got {MLP_HIDDEN}")
if GDN2_RESIDUAL_RANK <= 0:
    raise ValueError(
        f"GDN2_RESIDUAL_RANK must be positive, got {GDN2_RESIDUAL_RANK}"
    )


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert files, f"no shards match {filename_pattern}"
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    # cycle so runs longer than the downloaded shard slice keep training.
    file_iter = itertools.cycle(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (norm(x.float()) * self.gains).type_as(x)

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class BiasFreeLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))

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
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class KimiDeltaAttention(nn.Module):
    """Training-only Kimi-K3 KDA wrapper around FLA's released chunk kernel."""

    def __init__(
        self,
        dim: int,
        head_dim: int = 128,
        num_heads: int = KDA_NUM_HEADS,
        conv_size: int = 4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = self.num_heads * head_dim

        self.q_proj = BiasFreeLinear(dim, self.projection_size)
        self.k_proj = BiasFreeLinear(dim, self.projection_size)
        self.v_proj = BiasFreeLinear(dim, self.projection_size)
        self.q_conv1d = ShortConvolution(
            hidden_size=self.projection_size,
            kernel_size=conv_size,
            bias=False,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=self.projection_size,
            kernel_size=conv_size,
            bias=False,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=self.projection_size,
            kernel_size=conv_size,
            bias=False,
            activation="silu",
        )

        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.f_a_proj = BiasFreeLinear(dim, head_dim)
        self.f_b_proj = BiasFreeLinear(head_dim, self.projection_size)
        self.dt_bias = nn.Parameter(torch.empty(self.projection_size, dtype=torch.float32))
        self.b_proj = BiasFreeLinear(dim, self.num_heads)
        self.g_a_proj = BiasFreeLinear(dim, head_dim)
        self.g_b_proj = BiasFreeLinear(head_dim, self.projection_size)
        self.o_norm = FusedRMSNormGated(head_dim, eps=1e-6, activation="sigmoid")
        self.o_proj = BiasFreeLinear(self.projection_size, dim)

    def forward(self, x: Tensor):
        B, T, _ = x.shape
        q, _ = self.q_conv1d(x=self.q_proj(x), output_final_state=False)
        k, _ = self.k_conv1d(x=self.k_proj(x), output_final_state=False)
        v, _ = self.v_conv1d(x=self.v_proj(x), output_final_state=False)

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        decay_logits = self.f_b_proj(self.f_a_proj(x)).view(
            B, T, self.num_heads, self.head_dim
        )
        beta_logits = self.b_proj(x).float()

        y, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=decay_logits,
            beta=beta_logits,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            state_v_first=DELTA_STATE_V_FIRST,
            disable_recompute=DELTA_DISABLE_RECOMPUTE,
        )
        output_gate = self.g_b_proj(self.g_a_proj(x)).view(
            B, T, self.num_heads, self.head_dim
        )
        y = self.o_norm(y, output_gate).reshape(B, T, self.projection_size)
        return self.o_proj(y)


class KDACenteredGDN2Attention(KimiDeltaAttention):
    """KDA plus a zero-initialized low-rank channel-wise erase residual."""

    def __init__(
        self,
        dim: int,
        head_dim: int = 128,
        num_heads: int = KDA_NUM_HEADS,
        conv_size: int = 4,
        residual_rank: int = GDN2_RESIDUAL_RANK,
    ):
        super().__init__(
            dim=dim,
            head_dim=head_dim,
            num_heads=num_heads,
            conv_size=conv_size,
        )
        self.erase_a_proj = BiasFreeLinear(dim, residual_rank)
        self.erase_b_proj = BiasFreeLinear(residual_rank, self.projection_size)

    def forward(self, x: Tensor):
        B, T, _ = x.shape
        q, _ = self.q_conv1d(x=self.q_proj(x), output_final_state=False)
        k, _ = self.k_conv1d(x=self.k_proj(x), output_final_state=False)
        v, _ = self.v_conv1d(x=self.v_proj(x), output_final_state=False)

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        decay_logits = self.f_b_proj(self.f_a_proj(x)).view(
            B, T, self.num_heads, self.head_dim
        )

        beta_logits = self.b_proj(x).float().unsqueeze(-1)
        erase_residual = self.erase_b_proj(self.erase_a_proj(x)).float().view(
            B, T, self.num_heads, self.head_dim
        )
        erase_gate = (beta_logits + erase_residual).sigmoid().to(v.dtype)
        write_gate = beta_logits.sigmoid().to(v.dtype).expand_as(v)

        y, _ = chunk_gdn2(
            q=q,
            k=k,
            v=v,
            g=decay_logits,
            b=erase_gate,
            w=write_gate,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            state_v_first=DELTA_STATE_V_FIRST,
            disable_recompute=DELTA_DISABLE_RECOMPUTE,
        )
        output_gate = self.g_b_proj(self.g_a_proj(x)).view(
            B, T, self.num_heads, self.head_dim
        )
        y = self.o_norm(y, output_gate).reshape(B, T, self.projection_size)
        return self.o_proj(y)


class GatedDeltaNet2Attention(nn.Module):
    """Full GDN-2 wrapper using FLA's chunkwise training kernel."""

    def __init__(
        self,
        dim: int,
        head_dim: int = 128,
        num_heads: int = KDA_NUM_HEADS,
        conv_size: int = 4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = self.num_heads * self.head_dim

        self.q_proj = BiasFreeLinear(dim, self.projection_size)
        self.k_proj = BiasFreeLinear(dim, self.projection_size)
        self.v_proj = BiasFreeLinear(dim, self.projection_size)
        self.q_conv1d = ShortConvolution(
            hidden_size=self.projection_size,
            kernel_size=conv_size,
            bias=False,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=self.projection_size,
            kernel_size=conv_size,
            bias=False,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=self.projection_size,
            kernel_size=conv_size,
            bias=False,
            activation="silu",
        )

        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.f_a_proj = BiasFreeLinear(dim, head_dim)
        self.f_b_proj = BiasFreeLinear(head_dim, self.projection_size)
        self.dt_bias = nn.Parameter(torch.empty(self.projection_size, dtype=torch.float32))
        self.b_proj = BiasFreeLinear(dim, self.projection_size)
        self.w_proj = BiasFreeLinear(dim, self.projection_size)
        self.g_a_proj = BiasFreeLinear(dim, head_dim)
        self.g_b_proj = Linear(head_dim, self.projection_size)
        # The GDN-2 paper and NVIDIA reference use an SiLU output gate.
        self.o_norm = FusedRMSNormSwishGate(head_dim, eps=1e-5)
        self.o_proj = BiasFreeLinear(self.projection_size, dim)

    def forward(self, x: Tensor):
        B, T, _ = x.shape
        q, _ = self.q_conv1d(x=self.q_proj(x), output_final_state=False)
        k, _ = self.k_conv1d(x=self.k_proj(x), output_final_state=False)
        v, _ = self.v_conv1d(x=self.v_proj(x), output_final_state=False)

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        decay_logits = self.f_b_proj(self.f_a_proj(x)).view(
            B, T, self.num_heads, self.head_dim
        )
        erase_gate = self.b_proj(x).sigmoid().view(
            B, T, self.num_heads, self.head_dim
        )
        write_gate = self.w_proj(x).sigmoid().view(
            B, T, self.num_heads, self.head_dim
        )

        y, _ = chunk_gdn2(
            q=q,
            k=k,
            v=v,
            g=decay_logits,
            b=erase_gate,
            w=write_gate,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=False,
            state_v_first=DELTA_STATE_V_FIRST,
            disable_recompute=DELTA_DISABLE_RECOMPUTE,
        )
        output_gate = self.g_b_proj(self.g_a_proj(x)).view(
            B, T, self.num_heads, self.head_dim
        )
        y = self.o_norm(y, output_gate).reshape(B, T, self.projection_size)
        return self.o_proj(y)

# Each FLA primitive deliberately graph-breaks under torch.compile. Treating
# the complete delta-attention module as one eager region can avoid repeatedly
# leaving and re-entering compiled code around its convolutions, recurrence,
# and gated norm. Keep this opt-in until the whole-step benchmark confirms it.
if DELTA_EAGER_MODULE:
    KimiDeltaAttention.forward = torch.compiler.disable(KimiDeltaAttention.forward)
    KDACenteredGDN2Attention.forward = torch.compiler.disable(
        KDACenteredGDN2Attention.forward
    )
    GatedDeltaNet2Attention.forward = torch.compiler.disable(
        GatedDeltaNet2Attention.forward
    )

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = MLP_HIDDEN
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, use_kda: bool):
        super().__init__()
        self.use_kda = use_kda
        if use_kda:
            attention_types = {
                "kda": KimiDeltaAttention,
                "gdn2": GatedDeltaNet2Attention,
                "gdn2_kda_erase": KDACenteredGDN2Attention,
            }
            self.attn = attention_types[DELTA_ATTENTION_TYPE](dim)
        else:
            self.attn = CausalSelfAttention(dim)
        self.norm1 = RMSNorm(dim)
        self.use_mlp = not use_kda or DELTA_MLP_ON_DELTA
        if self.use_mlp:
            self.mlp = MLP(dim)
            self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        if self.use_mlp:
            x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        # Sequential lets Dynamo statically inline the fixed-depth stack.
        # A Python loop containing an eager FLA region makes Dynamo abandon
        # the whole GPT frame, leaving every surrounding MLP and norm eager.
        self.blocks = nn.Sequential(*[
            Block(model_dim, use_kda=layer_idx in DELTA_LAYER_INDICES)
            for layer_idx in range(num_layers)
        ])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        x = self.blocks(x)
        return self.compute_loss(x, targets)

    def compute_loss(self, x: Tensor, targets: Tensor):
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

@torch.no_grad()
def initialize_model(model: GPT, seed: int):
    """Pair the unchanged trunk with the baseline and initialize KDA safely."""

    torch.manual_seed(seed)
    for p in model.parameters():
        p.fill_(float("nan"))

    def normal_weight(weight: Tensor, generator=None):
        std = 0.33**0.5 / weight.size(-1)**0.5
        weight.normal_(std=std, generator=generator)

    def paired_prefix_weight(
        weight: Tensor,
        baseline_shape: tuple[int, int],
        reference_generator=None,
        extra_generator=None,
    ):
        reference = weight.new_empty(baseline_shape)
        normal_weight(reference, generator=reference_generator)
        if weight.size(1) > reference.size(1):
            raise ValueError("Paired weight cannot be wider than its baseline")
        paired_rows = min(weight.size(0), reference.size(0))
        weight[:paired_rows].copy_(reference[:paired_rows, :weight.size(1)])
        if paired_rows < weight.size(0):
            if extra_generator is None:
                raise ValueError("Expanded paired weight requires an extra generator")
            normal_weight(weight[paired_rows:], generator=extra_generator)

    def zero_linear(linear: nn.Linear):
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.zero_()

    model.embed.weight.normal_()
    for layer_idx, block in enumerate(model.blocks):
        if isinstance(block.attn, (KimiDeltaAttention, GatedDeltaNet2Attention)):
            attn = block.attn
            local_generator = torch.Generator(device=attn.q_proj.weight.device)
            local_generator.manual_seed(seed + 10_000 + layer_idx)

            # Delta blocks stay entirely off the shared RNG so the six
            # retained dense blocks remain exactly paired with the original
            # six-layer baseline. This also keeps delta attention identical
            # when ablating attention-only mixers against full delta blocks.
            baseline_attn_shape = (attn.q_proj.in_features, attn.q_proj.in_features)
            paired_prefix_weight(
                attn.q_proj.weight,
                baseline_attn_shape,
                reference_generator=local_generator,
            )
            paired_prefix_weight(
                attn.k_proj.weight,
                baseline_attn_shape,
                reference_generator=local_generator,
            )
            paired_prefix_weight(
                attn.v_proj.weight,
                baseline_attn_shape,
                reference_generator=local_generator,
            )
            special_linears = (
                (
                    attn.f_a_proj,
                    attn.f_b_proj,
                    attn.b_proj,
                    attn.g_a_proj,
                    attn.g_b_proj,
                )
                if isinstance(attn, KimiDeltaAttention)
                else (
                    attn.f_a_proj,
                    attn.f_b_proj,
                    attn.b_proj,
                    attn.w_proj,
                    attn.g_a_proj,
                    attn.g_b_proj,
                )
            )
            for linear in special_linears:
                normal_weight(linear.weight, generator=local_generator)
                if linear.bias is not None:
                    linear.bias.zero_()
            if isinstance(attn, KDACenteredGDN2Attention):
                # Keep every inherited KDA draw identical, then initialize the
                # adapter input normally and its output at zero. The model
                # therefore starts on KDA's tied scalar-gate subspace.
                normal_weight(
                    attn.erase_a_proj.weight,
                    generator=local_generator,
                )
                zero_linear(attn.erase_b_proj)

            # An identity causal convolution preserves the current-token
            # projection before the required SiLU at initialization.
            for conv in (attn.q_conv1d, attn.k_conv1d, attn.v_conv1d):
                conv.weight.zero_()
                conv.weight[:, 0, -1] = 1

            if isinstance(attn, GatedDeltaNet2Attention):
                # Match NVIDIA's GDN-2 reference: exp(A_log) starts uniformly
                # in [1, 16]. Keep this variant-only draw off the shared RNG.
                decay_rate_generator = torch.Generator(
                    device=attn.q_proj.weight.device
                )
                decay_rate_generator.manual_seed(seed + 20_000 + layer_idx)
                attn.A_log.copy_(
                    torch.empty_like(attn.A_log).uniform_(
                        1,
                        16,
                        generator=decay_rate_generator,
                    ).log()
                )
            else:
                # KDA retains its bounded safe-gate initialization at A=1.
                attn.A_log.zero_()
            # Replay the completed KDA reference's pre-dt RNG draws so KDA
            # and GDN-2 start from identical decay time constants even though
            # their variant-specific projection sets differ.
            decay_generator = torch.Generator(device=attn.q_proj.weight.device)
            decay_generator.manual_seed(seed + 10_000 + layer_idx)
            for shape in (
                (attn.head_dim, attn.q_proj.in_features),
                (attn.projection_size, attn.head_dim),
                (KDA_NUM_HEADS, attn.q_proj.in_features),
                (attn.projection_size, attn.q_proj.in_features),
            ):
                reference_draw = attn.q_proj.weight.new_empty(shape)
                normal_weight(reference_draw, generator=decay_generator)
            log_dt = torch.empty_like(attn.dt_bias).uniform_(
                math.log(0.001),
                math.log(0.1),
                generator=decay_generator,
            )
            dt = log_dt.exp().clamp(min=1e-4)
            attn.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            attn.o_norm.weight.fill_(1)
            zero_linear(attn.o_proj)
        else:
            attn = block.attn
            for linear in (attn.q, attn.k, attn.v):
                normal_weight(linear.weight)
                linear.bias.zero_()
            zero_linear(attn.proj)

        if block.use_kda:
            block.norm1.gains.fill_(1)
        else:
            block.norm1.gains.normal_(mean=1, std=0)
        if block.use_mlp:
            baseline_mlp_shape = (
                4 * block.mlp.fc.in_features,
                block.mlp.fc.in_features,
            )
            mlp_extra_generator = torch.Generator(
                device=block.mlp.fc.weight.device
            )
            mlp_extra_generator.manual_seed(seed + 30_000 + layer_idx)
            paired_prefix_weight(
                block.mlp.fc.weight,
                baseline_mlp_shape,
                reference_generator=(
                    mlp_extra_generator if block.use_kda else None
                ),
                extra_generator=mlp_extra_generator,
            )
            block.mlp.fc.bias.zero_()
            zero_linear(block.mlp.proj)
            if block.use_kda:
                block.norm2.gains.fill_(1)
            else:
                block.norm2.gains.normal_(mean=1, std=0)

    zero_linear(model.proj)
    model.norm1.gains.normal_(mean=1, std=0)
    model.norm2.gains.normal_(mean=1, std=0)

    uninitialized = [name for name, p in model.named_parameters() if not p.isfinite().all()]
    if uninitialized:
        raise RuntimeError(f"Uninitialized parameters: {uninitialized}")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


########################################
#                Setup                 #
########################################

# Single-process shim: default the torchrun env vars so `python3` launches work.
if "RANK" not in os.environ:
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29647")

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
delta_label = {
    "kda": "KDA",
    "gdn2": "GDN2",
    "gdn2_kda_erase": "KDA-centered GDN2 erase",
}[DELTA_ATTENTION_TYPE]
attention_schedule = ",".join(
    delta_label if layer_idx in DELTA_LAYER_INDICES else "dense"
    for layer_idx in range(NUM_LAYERS)
)
print0(
    f"Using fla-core {fla.__version__}; attention schedule: "
    f"{attention_schedule}"
)
print0(f"{delta_label} disable_recompute: {DELTA_DISABLE_RECOMPUTE}")
print0(f"{delta_label} state_v_first: {DELTA_STATE_V_FIRST}")
print0(f"{delta_label} eager module boundary: {DELTA_EAGER_MODULE}")
print0(f"{delta_label} blockwise compile: {DELTA_BLOCKWISE_COMPILE}")
print0(f"{delta_label} layers: {sorted(DELTA_LAYER_INDICES)}")
print0(f"compile mode: {DELTA_COMPILE_MODE}")
print0(f"CUDA graphs with persistent grad buffers: {DELTA_USE_CUDAGRAPHS}")
print0(f"{delta_label} blocks include MLPs: {DELTA_MLP_ON_DELTA}")
print0(f"MLP hidden width: {MLP_HIDDEN}")
if DELTA_ATTENTION_TYPE == "gdn2_kda_erase":
    print0(f"GDN2 erase residual rank: {GDN2_RESIDUAL_RANK}")
print0("="*100)

data_path = os.environ.get("DATA_PATH", "data/datasets/fineweb10B_gpt2")

val_tokens = int(os.environ.get("VAL_TOKENS", 64 * 524288))
batch_size = 8 * 64 * 1024
# SEQ_LEN reshapes the same token budget into longer rows (RoPE positions seen
# in pretraining bound the usable RL context). Halve MBS when doubling SEQ_LEN
# to keep microbatch tokens (and the 50304-wide logit buffer) constant.
seq_len = int(os.environ.get("SEQ_LEN", 1024))
mbs = int(os.environ.get("MBS", 8))
assert batch_size % (mbs * seq_len) == 0 and val_tokens % (mbs * seq_len) == 0
local_microbatches_per_step = batch_size // (
    dist.get_world_size() * mbs * seq_len
)
print0(
    f"batch: global_tokens={batch_size}, sequence_length={seq_len}, "
    f"microbatch_sequences={mbs}, "
    f"local_microbatches_per_step={local_microbatches_per_step}"
)
val_inputs, val_targets = next(distributed_data_generator(
    f"{data_path}/fineweb_val_*.bin", val_tokens, seq_len=seq_len))

# Challenge BPB metric with GPT-2 byte accounting: each GPT-2 BPE token maps
# to a fixed byte string, so a per-token byte-length LUT is exact.
gpt2_byte_lut = torch.load("data/tokenizers/gpt2_byte_lut.pt", weights_only=True).to(device)
assert gpt2_byte_lut.numel() == VOCAB_SIZE
with torch.no_grad():
    val_byte_count = float(gpt2_byte_lut[val_targets.reshape(-1)].to(torch.int64).sum())
    assert val_byte_count > 0

model = GPT(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, model_dim=512).cuda()
if DELTA_BLOCKWISE_COMPILE:
    # Compile every residual block independently so an opaque FLA recurrence
    # cannot make Dynamo abandon the entire fixed-depth model. Compile the
    # vocabulary projection and cross-entropy separately; they dominate the
    # non-block work and benefit heavily from fusion.
    for block in model.blocks:
        block.compile(dynamic=False, mode=DELTA_COMPILE_MODE)
    model.compute_loss = torch.compile(
        model.compute_loss,
        dynamic=False,
        mode=DELTA_COMPILE_MODE,
    )
else:
    model.compile(dynamic=False, mode=DELTA_COMPILE_MODE)
print0(f"parameters: {sum(p.numel() for p in model.parameters()):,}", console=True)
print0(f"val window: {val_tokens:,} tokens = {val_byte_count:,.0f} bytes "
       f"({val_byte_count/val_tokens:.3f} bytes/token)", console=True)

@torch.no_grad()
def reset_model_grads(model: nn.Module) -> None:
    """Clear gradients without aliasing CUDA-graph-owned backward outputs."""

    if not DELTA_USE_CUDAGRAPHS:
        model.zero_grad(set_to_none=True)
        return

    # AccumulateGrad may adopt a compiled backward output when ``.grad`` is
    # None. CUDA Graph Trees reuse that output storage on the next replay,
    # overwriting the accumulated value. Allocate each destination outside the
    # graph once, then preserve its address for the entire run.
    grads_to_clear = []
    for parameter in model.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(
                parameter,
                memory_format=torch.preserve_format,
            )
        else:
            grads_to_clear.append(parameter.grad)
    if grads_to_clear:
        torch._foreach_zero_(grads_to_clear)


def mark_model_step_begin() -> None:
    """Delimit one top-level compiled forward/backward or inference call."""

    if DELTA_USE_CUDAGRAPHS:
        torch.compiler.cudagraph_mark_step_begin()


def run_compile_diagnostics(model: GPT, inputs: Tensor, targets: Tensor) -> None:
    """Record graph fragmentation and its warmed CPU dispatch upper bound."""

    from torch._dynamo.utils import counters
    from torch.profiler import ProfilerActivity, profile

    model.train()
    reset_model_grads(model)
    counters.clear()

    # Compile and autotune one real training microbatch before profiling it.
    mark_model_step_begin()
    loss = model(inputs, targets)
    loss.backward()
    torch.cuda.synchronize()
    reset_model_grads(model)

    dynamo_counters = {
        category: {str(key): int(value) for key, value in values.items()}
        for category, values in counters.items()
        if values
    }

    # Measure a separate synchronized iteration without profiler overhead.
    torch.cuda.synchronize()
    unprofiled_wall_start = time.perf_counter()
    mark_model_step_begin()
    loss = model(inputs, targets)
    loss.backward()
    torch.cuda.synchronize()
    unprofiled_wall_ms = 1000 * (time.perf_counter() - unprofiled_wall_start)
    reset_model_grads(model)

    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        mark_model_step_begin()
        loss = model(inputs, targets)
        loss.backward()
    torch.cuda.synchronize()
    reset_model_grads(model)

    cache_lookup_count = 0
    cache_lookup_self_cpu_us = 0.0
    compiled_region_count = 0
    compiled_region_self_cpu_us = 0.0
    compiled_region_keys = {}
    for event in prof.key_averages():
        if event.key == "TorchDynamo Cache Lookup":
            cache_lookup_count += event.count
            cache_lookup_self_cpu_us += event.self_cpu_time_total
        if event.key.startswith("Torch-Compiled Region"):
            compiled_region_count += event.count
            compiled_region_self_cpu_us += event.self_cpu_time_total
            compiled_region_keys[event.key] = {
                "count": event.count,
                "self_cpu_time_us": event.self_cpu_time_total,
            }

    microbatches_per_step = batch_size // (
        dist.get_world_size() * mbs * seq_len
    )
    profile_summary = {
        "microbatch_shape": list(inputs.shape),
        "microbatches_per_local_step": microbatches_per_step,
        "warmed_unprofiled_microbatch_wall_ms": unprofiled_wall_ms,
        "dynamo_counters_after_initial_compile": dynamo_counters,
        "torchdynamo_cache_lookup": {
            "count_per_microbatch": cache_lookup_count,
            "self_cpu_time_us_per_microbatch": cache_lookup_self_cpu_us,
            "estimated_count_per_local_step": (
                cache_lookup_count * microbatches_per_step
            ),
            "estimated_self_cpu_time_ms_per_local_step": (
                cache_lookup_self_cpu_us * microbatches_per_step / 1000
            ),
        },
        "compiled_regions": {
            "count_per_microbatch": compiled_region_count,
            "self_cpu_time_us_per_microbatch": compiled_region_self_cpu_us,
            "estimated_count_per_local_step": (
                compiled_region_count * microbatches_per_step
            ),
            "estimated_self_cpu_time_ms_per_local_step": (
                compiled_region_self_cpu_us * microbatches_per_step / 1000
            ),
            "keys": compiled_region_keys,
        },
    }
    output_dir = Path("ablation_results") / run_id
    output_path = output_dir / "compile_diagnostics.json"
    if dist.get_rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(profile_summary, indent=2, sort_keys=True) + "\n"
        )
        temporary_path.replace(output_path)
    graph_breaks = sum(dynamo_counters.get("graph_break", {}).values())
    print0(
        f"compile diagnostics saved to {output_path}: "
        f"graph_breaks={graph_breaks}, "
        f"compiled_regions={compiled_region_count}, "
        f"region_self_cpu_us={compiled_region_self_cpu_us:.1f}, "
        f"cache_self_cpu_us={cache_lookup_self_cpu_us:.1f}",
        console=True,
    )


num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for trial in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    train_steps = int(os.environ.get("ITERATIONS", 1000))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 20))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))

    # Pair all unchanged tensors with the baseline seed while giving KDA's
    # special decay, convolution, and gated-normalization parameters their
    # intended initialization.
    seed = int(os.environ.get("SEED", 1337))
    initialize_model(model, seed)
    reset_model_grads(model)
    if KDA_COMPILE_DIAGNOSTICS:
        run_compile_diagnostics(model, val_inputs[:mbs], val_targets[:mbs])

    # create the optimizer(s)
    embed_lr = float(os.environ.get("EMBED_LR", 0.7))
    proj_lr = float(os.environ.get("PROJ_LR", 0.004))
    delta_conv_lr = float(os.environ.get("KDA_CONV_LR", 0.004))
    delta_attention_types = (KimiDeltaAttention, GatedDeltaNet2Attention)
    delta_conv_params = [
        conv.weight
        for block in model.blocks
        if isinstance(block.attn, delta_attention_types)
        for conv in (block.attn.q_conv1d, block.attn.k_conv1d, block.attn.v_conv1d)
    ]
    delta_conv_param_ids = {id(p) for p in delta_conv_params}
    delta_decay_params = [
        p
        for block in model.blocks
        if isinstance(block.attn, delta_attention_types)
        for p in (block.attn.A_log, block.attn.dt_bias)
    ]
    delta_decay_param_ids = {id(p) for p in delta_decay_params}
    scalar_params = [
        p for p in model.parameters()
        if p.ndim < 2 and id(p) not in delta_decay_param_ids
    ]
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=embed_lr),
                        dict(params=[model.proj.weight], lr=proj_lr),
                        dict(params=scalar_params, lr=0.015),
                        dict(params=delta_conv_params, lr=delta_conv_lr),
                        dict(params=delta_decay_params, lr=0.015, weight_decay=0.0)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
    optimizer2 = Muon([
        p for p in model.blocks.parameters()
        if p.ndim >= 2 and id(p) not in delta_conv_param_ids
    ],
                      lr=0.025, weight_decay=0.05)
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta


    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator(
        f"{data_path}/fineweb_train_*.bin", batch_size, seq_len=seq_len)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    # start the clock
    training_time = 0
    last_val_step = 0
    dist.barrier()
    t0 = time.perf_counter()
    for step in range(train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        if step == train_steps or step % val_loss_every == 0:
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    mark_model_step_begin()
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= val_tokens
            val_bpb = (float(val_loss) / math.log(2.0)) * (val_tokens / val_byte_count)
            peak_vram_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
            peak_vram_reserved_mib = torch.cuda.max_memory_reserved() / 2**20
            process_peak_rss_mib = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            )
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} val_bpb:{val_bpb:.4f}"
                   + f" train_time:{1000*training_time:.0f}ms step_avg:{1000*step_avg:.2f}ms"
                   + f" peak_vram_allocated_mib:{peak_vram_allocated_mib:.0f}"
                   + f" peak_vram_reserved_mib:{peak_vram_reserved_mib:.0f}"
                   + f" process_peak_rss_mib:{process_peak_rss_mib:.0f}",
                   console=True)
            model.train()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0
        train_loss_sum = torch.zeros((), device=device)
        for i in range(len(inputs) // mbs):
            mark_model_step_begin()
            loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
            train_loss_sum += loss.detach()
            loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        # set optimization hyperparameters and take a step
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        reset_model_grads(model)
        approx_training_time = training_time + (time.perf_counter() - t0)
        if (step + 1) % train_log_every == 0:
            train_loss = float(train_loss_sum) / targets.numel()
            print0(f"step:{step+1}/{train_steps} train_loss:{train_loss:.4f}"
                   + f" train_time:{1000*approx_training_time:.0f}ms"
                   + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True)

    # self-describing checkpoint for postraining/model_io.py (off the clock)
    if dist.get_rank() == 0:
        suffix = f"_trial{trial}" if num_trials > 1 else ""
        ckpt_path = f"logs/{run_id}{suffix}_final_model.pt"
        torch.save({
            "model": model.state_dict(),
            "model_config": dict(
                vocab_size=VOCAB_SIZE,
                num_layers=NUM_LAYERS,
                model_dim=512,
                mlp_hidden=MLP_HIDDEN,
                delta_num_heads=KDA_NUM_HEADS,
                delta_attention_type=DELTA_ATTENTION_TYPE,
                delta_layer_indices=sorted(DELTA_LAYER_INDICES),
                delta_mlp_on_delta=DELTA_MLP_ON_DELTA,
                delta_residual_rank=(
                    GDN2_RESIDUAL_RANK
                    if DELTA_ATTENTION_TYPE == "gdn2_kda_erase"
                    else None
                ),
            ),
            "architecture": (
                f"nanogpt_mini_gpt2vocab_{DELTA_ATTENTION_TYPE}_"
                + "".join(
                    "k" if layer_idx in DELTA_LAYER_INDICES else "d"
                    for layer_idx in range(NUM_LAYERS)
                )
                + (
                    "_fullblocks_v3"
                    if DELTA_MLP_ON_DELTA
                    else "_mixers_v3"
                )
            ),
            "training_config": {
                "delta_disable_recompute": DELTA_DISABLE_RECOMPUTE,
                "delta_state_v_first": DELTA_STATE_V_FIRST,
                "delta_eager_module": DELTA_EAGER_MODULE,
                "delta_blockwise_compile": DELTA_BLOCKWISE_COMPILE,
                "delta_compile_mode": DELTA_COMPILE_MODE,
                "delta_use_cudagraphs": DELTA_USE_CUDAGRAPHS,
                "delta_mlp_on_delta": DELTA_MLP_ON_DELTA,
                "global_batch_tokens": batch_size,
                "microbatch_sequences": mbs,
                "local_microbatches_per_step": local_microbatches_per_step,
            },
            # RoPE positions seen in pretraining bound the usable RL context
            # (half-truncate rotary has no extrapolation).
            "train_seq_len": seq_len,
        }, ckpt_path)
        print0(f"saved checkpoint: {ckpt_path}", console=True)

dist.destroy_process_group()

"""Hierarchical byte language models on the measured KDA8 trunk.

The global model is the exact eight-layer ``KKKDKKKD`` trunk of the GPT-2
token baseline (`nanogpt_mini_gpt2vocab_kda_3to1_pm_train.py` with
``NUM_LAYERS=8 DELTA_LAYER_INDICES=0,1,2,4,5,6 MLP_HIDDEN=2070 NOPE=1``): KDA
mixers without MLPs, and two NoPE softmax-attention blocks with ReLU^2 MLPs.
Keeping its structure and initialization distributions identical means a
byte model and the token control differ only in how atoms reach and leave
the trunk. (RNG streams differ from the token trainer, so runs are not
seed-paired with it.)

Byte models wrap the trunk BLT-style:

    bytes -> byte + hashed n-gram embeddings -> local encoder (windowed attn)
          -> pool the encoder state at each patch end -> trunk over patches
          -> route each byte to the newest *completed* patch -> local decoder
          -> 257-way byte head (256 bytes + end-of-text)

Patchers: ``static`` (fixed stride); ``space`` -- SpaceByte's word
boundaries made causal (`spacelike_ends`), within the same chunk budget and
cap as the gated patchers; or ``entropy`` -- a learned causal gate.
The encoder carries a local next-byte head trained by its own CE; a patch
ends after byte ``t`` when that head's entropy for byte ``t + 1`` exceeds a
threshold, so hard bytes are decoded from a fresh trunk state. The threshold
tracks either a target mean patch length (``rate``) or the running mean
entropy (``mean``: continue while the next byte looks easier than average).
It is a buffer updated only from past batches, so boundaries are a
deterministic function of bytes ``<= t`` and the codelength stays valid.

``learned``: a continue/stop gate trained by the main loss (dynamic chunking
after H-Net, Hwang et al. 2025, made causal). At byte ``t`` a router scores
``p_t = (1 - cos(q(h_t), k(h_{t-1}))) / 2`` -- how much the byte state moved
-- and the chunk ends after ``t`` when ``p_t`` exceeds a threshold: 0.5 as in
H-Net (``fixed``), or, by default, the entropy patcher's ``rate`` controller,
which holds the realized chunk rate at ``1 / target_patch``. (With ``fixed``,
the ratio loss alone let the rate drift from 4.2 to 2.9 bytes per chunk
over a 1k-step run until the trunk budget bound it; NOTES 2026-09-30.) The hard decision gets
gradient two ways: trunk outputs are smoothed across chunks,
``z_k <- p_k z_k + (1 - p_k) z_{k-1}``, so ``p`` at a chunk end weighs the
new chunk against carrying the last one; and the trunk's contribution to
each byte is scaled by a straight-through confidence ``c_t`` (``p_t`` if it
ended a chunk, else ``1 - p_t``), applied as ``x + (c - sg(c)) * x`` so the
forward value is exactly unchanged. ``x`` is the routed trunk state (attention
decoder) or the NextLat start state lifted from it, not the whole state,
which the readout norm would rescale. (At a patch end the NextLat state is
that lift alone, so there the gradient comes only through smoothing.) H-Net's
ratio loss, with its hard rate F measured at 0.5 as in H-Net, keeps ``p``
calibrated so that the 0.5-crossing rate sits near ``1 / target_patch``; the
controller only corrects the residual drift. (Measured on the realized ends,
F would sit at the target under the controller, where the loss has no
gradient: p then saturated near 1, the boundaries became noise and the
decoder stopped using the trunk.) The trunk runs
a fixed budget of chunk slots per row (`budgeted_ends`), so every shape is
static.

Decoders: ``attention`` (windowed transformer with the routed trunk state
injected per block) or ``nextlat`` -- a NextLat-style recurrent unembedder:
the fresh state ``s = lift(trunk_k)`` at each patch end, then
``s <- s + MLP(norm([a_t, s]))`` for each byte consumed inside the patch,
with the encoder state of the consumed byte as the action ``a_t``.
``trunk_logits`` adds per-offset logits read straight from the routed trunk
state to the attention decoder's, so the 512-wide trunk state reaches the
byte softmax without passing through the 256-wide local stream.

Causality contract: the decoder state at byte ``t`` scores byte ``t + 1``.
It may read trunk output ``k`` only if patch ``k`` ends at or before ``t``;
patch ends must be a causal function of bytes ``<= t``. `PatchLayout`
encodes this once, and both static and learned patchers must produce one.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

# fla dispatches the KDA chunked backward to its TileLang backend by default;
# here TileLang's vendored TVM and the standalone tvm_ffi both register
# ``__ffi_repr__`` for one type index, so the first backward raises and the
# process then hangs on exit (see postraining/kda_backbone.py). fla caches
# the capability check when its KDA backends are imported, so pin the Triton
# reference backward before that import and verify it took.
os.environ.setdefault("FLA_TILELANG", "0")

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.backends.tilelang import KDATileLangBackend
except ImportError as exc:  # pragma: no cover - environment contract
    raise RuntimeError("pretraining.hbyte requires fla-core==0.5.2") from exc

if KDATileLangBackend.can_use():
    raise RuntimeError(
        "fla resolved the TileLang KDA backend, whose backward cannot run here; "
        f"FLA_TILELANG must be 0 before fla is first imported (it is {os.environ.get('FLA_TILELANG')!r})"
    )

BYTE_EOT = 256
BYTE_VOCAB = 257  # 256 byte values plus end-of-text
GPT2_VOCAB = 50_304
GPT2_EOT = 50_256
SOFTCAP = 15.0


@dataclass(frozen=True)
class TrunkConfig:
    dim: int = 512
    layers: int = 8
    kda_layers: tuple[int, ...] = (0, 1, 2, 4, 5, 6)
    kda_heads: int = 3
    head_dim: int = 128
    mlp_hidden: int = 2070
    nope: bool = True


@dataclass(frozen=True)
class LocalConfig:
    dim: int = 512
    encoder_layers: int = 1
    decoder_layers: int = 2
    head_dim: int = 128
    mlp_mult: int = 4
    window: int = 512
    # The encoder's attention window; 0 means ``window``. The decoder needs
    # long exact access to recent bytes, the encoder may not.
    encoder_window: int = 0
    # Token mixer per layer: "a" sliding-window attention over ``window``
    # bytes, "k" KDA over the whole row (as H-Net's Mamba-2 local layers).
    # Empty means all attention.
    encoder_mixers: str = ""
    decoder_mixers: str = ""
    kda_heads: int = 2

    def stage_window(self, stage: str) -> int:
        return (self.encoder_window or self.window) if stage == "encoder" else self.window

    def mixers(self, stage: str) -> str:
        layers = self.encoder_layers if stage == "encoder" else self.decoder_layers
        return getattr(self, f"{stage}_mixers") or "a" * layers

    def __post_init__(self):
        if self.window < 1 or self.encoder_window < 0:
            raise ValueError("window must be >= 1 and encoder_window >= 0")
        for stage, layers in (("encoder", self.encoder_layers), ("decoder", self.decoder_layers)):
            spec = getattr(self, f"{stage}_mixers")
            if spec and (len(spec) != layers or set(spec) - {"a", "k"}):
                raise ValueError(f"{stage}_mixers needs one of 'a' or 'k' per layer ({layers})")


@dataclass(frozen=True)
class HashConfig:
    orders: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    buckets: int = 65_536


@dataclass(frozen=True)
class ModelConfig:
    atom: str = "byte"  # "byte" or "gpt2"
    seq_len: int = 8192  # atoms per row
    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    local: LocalConfig = field(default_factory=LocalConfig)
    hashing: HashConfig = field(default_factory=HashConfig)
    patch_size: int = 4  # static stride
    patcher: str = "static"  # "static" | "space" | "entropy" | "learned"
    # Gate threshold: "rate" (both gates), "mean" (entropy), "fixed" (learned: 0.5).
    threshold_mode: str = "rate"
    target_patch: float = 4.0  # dynamic patchers: mean bytes per patch (sizes the chunk budget)
    max_patch: int = 8  # dynamic patchers: hard cap on bytes per patch
    aux_weight: float = 1.0  # entropy patcher: weight of the local next-byte CE
    aux_detach: bool = False  # entropy patcher: keep its CE out of the encoder
    decoder: str = "attention"  # "attention" | "nextlat"
    nextlat_proj_factor: float = 1.6
    nextlat_local_reset: bool = False  # add W h_end to the state at each patch end
    patch_pool: str = "end"  # trunk input per patch: "end" state or "attn" pooling
    pool_heads: int = 4
    # Learned patchers: static trunk length per row, as a multiple of the
    # nominal ``seq_len / target_patch`` chunks (see `budgeted_ends`).
    chunk_slack: float = 1.25
    ratio_weight: float = 0.03  # learned patcher: weight of the chunk-rate loss
    # Attention decoder: zero-init logits read straight from the routed
    # trunk state (one 512 -> 257 map per offset bucket 0, 1, 2, 3+), so the
    # trunk's width reaches the byte softmax past the local residual stream.
    trunk_logits: bool = False
    # Block-anchored prefix memory (`block_prefix_ids`), 2**prefix_bits hashed
    # rows per table (0 = off): the completed block's string feeds the trunk
    # input, and the current block's prefix feeds the decoder input and
    # modulates a 512-wide trunk readout into the byte logits. Each path
    # starts at zero, so a fresh model is the model without it.
    prefix_bits: int = 0
    # Block n-gram memory: the trunk input also reads hashed embeddings of the
    # last 2 .. block_ngram completed block strings (word bigrams, trigrams),
    # each from its own 2**prefix_bits table behind a zero-init gain.
    block_ngram: int = 1
    # Contextual prefix memory: the decoder also reads the current prefix
    # hashed together with the previous completed block (a hashed word-bigram
    # lexicon), as an input embedding and in the readout modulation.
    prefix_context: bool = False
    # The readout modulation also sees the decoder's final byte state,
    # projected to trunk width by a zero-init linear: the prefix then gates a
    # joint (trunk, local context) readout instead of the trunk state alone.
    prefix_local: bool = False
    # Engram-style context gate on each memory read added to a residual
    # input: sigmoid(<norm(state), norm(memory)> / sqrt(d)), so a hashed row
    # that disagrees with the state (a collision, a rare string) is damped.
    memory_gate: bool = False

    def __post_init__(self):
        if self.atom not in {"byte", "gpt2"}:
            raise ValueError(f"unknown atom {self.atom!r}")
        if self.patcher not in {"static", "space", "entropy", "learned"} or self.decoder not in {"attention", "nextlat"}:
            raise ValueError("unknown patcher or decoder")
        if self.patch_pool not in {"end", "attn"}:
            raise ValueError("patch_pool must be 'end' or 'attn'")
        modes = {"entropy": {"rate", "mean"}, "learned": {"rate", "fixed"}}.get(self.patcher, {"rate"})
        if self.threshold_mode not in modes:
            raise ValueError(f"the {self.patcher} patcher's threshold_mode must be one of {sorted(modes)}")
        if self.atom == "byte" and self.seq_len % self.patch_size:
            raise ValueError("static patching needs seq_len divisible by patch_size")
        if not 1 < self.target_patch <= self.max_patch:
            raise ValueError("need 1 < target_patch <= max_patch")
        if self.chunk_slack < 1:
            raise ValueError("chunk_slack must be >= 1")
        if self.trunk_logits and self.decoder != "attention":
            raise ValueError("trunk_logits needs the attention decoder")
        if self.prefix_bits and (self.decoder != "attention" or self.atom != "byte"):
            raise ValueError("prefix_bits needs a byte model with the attention decoder")
        # Up to 3 + (block_ngram - 1) + 2 dense (2**bits, <=512) tables with
        # Adam state: at 20 bits the unigram three alone are ~1.3B parameters.
        if not 0 <= self.prefix_bits <= 20:
            raise ValueError("prefix_bits must be in [0, 20]")
        if (self.block_ngram > 1 or self.prefix_context or self.memory_gate or self.prefix_local) and not self.prefix_bits:
            raise ValueError("block_ngram, prefix_context, prefix_local and memory_gate need prefix_bits")
        if not 1 <= self.block_ngram <= 4:
            raise ValueError("block_ngram must be in [1, 4]")

    @property
    def patch_cap(self) -> int:
        """Longest possible patch: bounds the recurrent decoder's unroll."""
        return self.patch_size if self.patcher == "static" else self.max_patch

    def chunk_budget(self, length: int) -> int:
        """Trunk sequence length for a row of ``length`` atoms: the static
        stride's patch count, or the learned patchers' fixed budget (a
        multiple of 64, the KDA chunk). One static shape per row length means
        one compiled trunk graph and no host sync to size it."""
        if self.patcher == "static":
            return length // self.patch_size
        # At least ``ceil(length / max_patch)``, the forced ends of a row with
        # no natural ones, since ``chunk_slack >= 1`` and ``target_patch <=
        # max_patch``.
        return min(length, 64 * math.ceil(length * self.chunk_slack / self.target_patch / 64))


# --------------------------------------------------------------------------
# Shared primitives (ported from the token trainer so init and numerics match)
# --------------------------------------------------------------------------


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def memory_read(gain: Tensor, memory: Tensor, state: Tensor, gated: bool) -> Tensor:
    """``gain * memory`` to add to ``state``; with ``gated``, scaled by the
    Engram gate ``sigmoid(<norm(state), norm(memory)> / sqrt(d))``."""
    memory = memory.type_as(state)
    if gated:
        agreement = (norm(state) * norm(memory)).sum(-1, keepdim=True) / math.sqrt(state.size(-1))
        memory = memory * torch.sigmoid(agreement)
    return (gain * memory).type_as(state)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return (norm(x.float()) * self.gains).type_as(x)


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        bias = None if self.bias is None else self.bias.type_as(x)
        return F.linear(x, self.weight.type_as(x), bias)


class Rotary(nn.Module):
    """Half-truncate RoPE, as in the token trainer."""

    def __init__(self, dim: int):
        super().__init__()
        freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer(
            "angular_freq", torch.cat([freq, freq.new_zeros(dim // 4)]), persistent=False
        )

    def forward(self, x_bthd: Tensor) -> Tensor:
        pos = torch.arange(x_bthd.size(1), dtype=torch.float32, device=x_bthd.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_bthd.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), 3).type_as(x_bthd)


class ReluSquaredMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc = Linear(dim, hidden)
        self.proj = Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.fc(x).relu().square())


def softcap(logits: Tensor) -> Tensor:
    return SOFTCAP * logits * (logits.square() + SOFTCAP**2).rsqrt()


# --------------------------------------------------------------------------
# Global trunk
# --------------------------------------------------------------------------


class KimiDeltaAttention(nn.Module):
    """KDA via FLA's chunk kernel; identical to the token trainer's module."""

    def __init__(self, dim: int, head_dim: int, num_heads: int, conv_size: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = num_heads * head_dim
        self.q_proj = Linear(dim, self.projection_size, bias=False)
        self.k_proj = Linear(dim, self.projection_size, bias=False)
        self.v_proj = Linear(dim, self.projection_size, bias=False)
        self.q_conv1d, self.k_conv1d, self.v_conv1d = (
            ShortConvolution(
                hidden_size=self.projection_size,
                kernel_size=conv_size,
                bias=False,
                activation="silu",
            )
            for _ in range(3)
        )
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.f_a_proj = Linear(dim, head_dim, bias=False)
        self.f_b_proj = Linear(head_dim, self.projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.projection_size, dtype=torch.float32))
        self.b_proj = Linear(dim, num_heads, bias=False)
        self.g_a_proj = Linear(dim, head_dim, bias=False)
        self.g_b_proj = Linear(head_dim, self.projection_size, bias=False)
        self.o_norm = FusedRMSNormGated(head_dim, eps=1e-6, activation="sigmoid")
        self.o_proj = Linear(self.projection_size, dim, bias=False)

    # FLA primitives graph-break under torch.compile; one eager region per
    # module avoids re-entering compiled code around every primitive.
    @torch.compiler.disable
    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        shape = (B, T, self.num_heads, self.head_dim)
        q, _ = self.q_conv1d(x=self.q_proj(x), output_final_state=False)
        k, _ = self.k_conv1d(x=self.k_proj(x), output_final_state=False)
        v, _ = self.v_conv1d(x=self.v_proj(x), output_final_state=False)
        y, _ = chunk_kda(
            q=q.view(shape),
            k=k.view(shape),
            v=v.view(shape),
            g=self.f_b_proj(self.f_a_proj(x)).view(shape),
            beta=self.b_proj(x).float(),
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            state_v_first=True,
            disable_recompute=True,
        )
        gate = self.g_b_proj(self.g_a_proj(x)).view(shape)
        return self.o_proj(self.o_norm(y, gate).reshape(B, T, self.projection_size))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, nope: bool):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.proj = Linear(dim, dim)
        self.rotary = None if nope else Rotary(head_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        shape = (B, T, self.num_heads, self.head_dim)
        q, k = norm(self.q(x).view(shape)), norm(self.k(x).view(shape))
        if self.rotary is not None:
            q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            self.v(x).view(shape).transpose(1, 2),
            scale=0.12,
            is_causal=True,
        )
        return self.proj(y.transpose(1, 2).reshape(B, T, -1))


class TrunkBlock(nn.Module):
    def __init__(self, config: TrunkConfig, use_kda: bool):
        super().__init__()
        self.use_kda = use_kda
        self.attn = (
            KimiDeltaAttention(config.dim, config.head_dim, config.kda_heads)
            if use_kda
            else CausalSelfAttention(config.dim, config.head_dim, config.nope)
        )
        self.norm1 = RMSNorm(config.dim)
        # The measured baseline puts MLPs only on the dense blocks.
        self.mlp = None if use_kda else ReluSquaredMLP(config.dim, config.mlp_hidden)
        self.norm2 = None if use_kda else RMSNorm(config.dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        return x


class Trunk(nn.Module):
    def __init__(self, config: TrunkConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.Sequential(
            *[TrunkBlock(config, i in config.kda_layers) for i in range(config.layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


# --------------------------------------------------------------------------
# Local byte stack
# --------------------------------------------------------------------------


def sliding_window_mask(seq_len: int, window: int, device) -> BlockMask:
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < window)

    return create_block_mask(mask_mod, None, None, seq_len, seq_len, device=device)


# Flex tiles for the local layers: on the RTX 5090 at 2 heads x 128, window
# 1024, 24 x 8192 bytes, fwd+bwd is 6.40 ms against 7.40 ms with the
# defaults (identical gradients).
LOCAL_FLEX_OPTIONS = dict(BLOCK_M1=64, BLOCK_N1=64, BLOCK_M2=64, BLOCK_N2=64, num_warps=8, num_stages=2)


class LocalAttention(nn.Module):
    """Causal sliding-window attention over bytes (RoPE, QK-norm)."""

    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.proj = Linear(dim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor, block_mask: BlockMask) -> Tensor:
        B, T, _ = x.shape
        shape = (B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(shape)
        # rms_norm runs in float32 under autocast; flex_attention's backward
        # needs q, k and v in one dtype.
        q = self.rotary(norm(self.q(x).view(shape))).type_as(v)
        k = self.rotary(norm(self.k(x).view(shape))).type_as(v)
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            scale=0.12,
            kernel_options=LOCAL_FLEX_OPTIONS,
        )
        return self.proj(y.transpose(1, 2).reshape(B, T, -1))


class LocalBlock(nn.Module):
    def __init__(self, config: LocalConfig, mixer: str = "a"):
        super().__init__()
        self.norm1 = RMSNorm(config.dim)
        self.attn = (
            LocalAttention(config.dim, config.head_dim)
            if mixer == "a"
            else KimiDeltaAttention(config.dim, config.head_dim, config.kda_heads)
        )
        self.norm2 = RMSNorm(config.dim)
        self.mlp = ReluSquaredMLP(config.dim, config.mlp_mult * config.dim)

    def forward(self, x: Tensor, block_mask: BlockMask) -> Tensor:
        h = self.norm1(x)
        x = x + (self.attn(h, block_mask) if isinstance(self.attn, LocalAttention) else self.attn(h))
        return x + self.mlp(self.norm2(x))


def hashed_ngram_ids(x: Tensor, orders: tuple[int, ...], buckets: int) -> Tensor:
    """Causal hashed n-gram ids ``(len(orders), B, T)``, offset per order.

    The n-gram for position ``t`` is ``x[t-n+1 .. t]``; positions before the
    row start read a pad symbol, so the ids depend only on bytes ``<= t``
    inside the row.
    """
    x = x.long()
    pad = torch.full_like(x[:, :1], BYTE_VOCAB)
    ids = []
    h = torch.zeros_like(x)
    max_order = max(orders)
    shifted = x
    for i in range(max_order):
        if i:
            shifted = torch.cat([pad.expand(-1, i), x[:, :-i]], dim=1)
        # FNV-style multiply-xor; int64 wraparound is intended.
        h = (h * 0x100000001B3) ^ (shifted + 1)
        n = i + 1
        if n in orders:
            mixed = h ^ (h >> 29)
            mixed = mixed * 0x5BD1E995
            ids.append(torch.remainder(mixed ^ (mixed >> 32), buckets) + len(ids) * buckets)
    return torch.stack(ids)


# --------------------------------------------------------------------------
# Patch layouts
# --------------------------------------------------------------------------


@dataclass
class PatchLayout:
    """Where patches end and which trunk output every byte may read.

    ``end_index[b, k]``: byte position of patch ``k``'s last byte (padded
    patches point at 0 and are masked by ``patch_valid``).
    ``route_index[b, t]``: index into ``[bos, trunk_0, trunk_1, ...]`` of the
    newest patch completed at or before ``t`` (0 means none yet).
    ``offset[b, t]``: 0 if the *target* byte ``t + 1`` is the first byte after
    a completed patch, i.e. the first byte decoded from a fresh trunk state.
    """

    end_index: Tensor
    patch_valid: Tensor
    route_index: Tensor
    offset: Tensor
    # `block_prefix_ids` at each byte, when the model has prefix memory.
    prefix_id: Tensor | None = None
    # The prefix hashed with the previous completed block (`prefix_context`).
    context_id: Tensor | None = None

    @staticmethod
    def static(batch: int, seq_len: int, patch_size: int, device) -> "PatchLayout":
        t = torch.arange(seq_len, device=device)
        ends = torch.arange(patch_size - 1, seq_len, patch_size, device=device)
        return PatchLayout(
            end_index=ends.expand(batch, -1),
            patch_valid=torch.ones(batch, ends.numel(), dtype=torch.bool, device=device),
            route_index=((t + 1) // patch_size).expand(batch, -1),
            offset=((t + 1) % patch_size).expand(batch, -1),
        )

    @staticmethod
    def from_ends(ends: Tensor, budget: int) -> "PatchLayout":
        """Layout from a causal (B, T) patch-end mask with at most ``budget``
        ends per row (`budgeted_ends`), in exactly ``budget`` trunk slots.
        Unused slots sit after every real patch, so the causal trunk never
        lets them influence a real one, and no byte routes to them."""
        B, T = ends.shape
        t = torch.arange(T, device=ends.device)
        route = ends.long().cumsum(1)
        # Non-ends scatter into a discarded extra column.
        slot = torch.where(ends, route - 1, budget)
        end_index = torch.zeros(B, budget + 1, dtype=torch.long, device=ends.device)
        end_index = end_index.scatter(1, slot, t.expand(B, -1))[:, :budget]
        last_end = torch.where(ends, t, -1).cummax(1).values
        return PatchLayout(
            end_index=end_index,
            patch_valid=torch.arange(budget, device=ends.device) < route[:, -1:],
            route_index=route,
            offset=t - last_end,
        )


def capped_ends(natural: Tensor, cap: int) -> Tensor:
    """Add forced ends so no patch exceeds ``cap`` bytes. Closed form: inside
    each run after a natural end, every ``cap``-th byte ends a patch."""
    t = torch.arange(natural.size(1), device=natural.device)
    since = t - torch.where(natural, t, -1).cummax(1).values  # 0 at natural ends
    return natural | ((since % cap == 0) & (since > 0))


def spacelike_ends(x: Tensor) -> Tensor:
    """SpaceByte's boundaries, made causal: a patch ends after byte ``t`` when
    ``t`` is the first spacelike byte of a run. Spacelike means anything but
    an ASCII letter or digit or a non-ASCII byte (end-of-text is spacelike).
    SpaceByte also counts UTF-8 lead bytes; ending *after* them would split
    every multi-byte character, so here they stay inside words (unspaced
    scripts then patch by the cap). SpaceByte's own rule also reads
    ``t + 1``; this one reads bytes ``t - 1`` and ``t`` only."""
    alnum = ((x >= 48) & (x <= 57)) | ((x >= 65) & (x <= 90)) | ((x >= 97) & (x <= 122))
    space = ~(alnum | ((x >= 0x80) & (x < BYTE_EOT)))
    # Row start: nothing before it, so a leading spacelike byte does not end.
    prev = torch.cat([torch.ones_like(space[:, :1]), space[:, :-1]], dim=1)
    return space & ~prev


PREFIX_BASE = 0x100000001B3  # odd, so invertible mod 2**64


def _signed64(v: int) -> int:
    v %= 1 << 64
    return v - (1 << 64) if v >= 1 << 63 else v


def prefix_powers(length: int, device) -> Tensor:
    """``(2, length)`` int64: ``PREFIX_BASE ** t`` and its inverse mod 2**64."""
    inverse = pow(PREFIX_BASE, -1, 1 << 64)
    pw, ipw, a, b = [], [], 1, 1
    for _ in range(length):
        pw.append(_signed64(a))
        ipw.append(_signed64(b))
        a, b = a * PREFIX_BASE % (1 << 64), b * inverse % (1 << 64)
    return torch.tensor([pw, ipw], dtype=torch.long, device=device)


# Odd multiplier that chains block hashes into n-gram hashes.
NGRAM_MULTIPLIER = _signed64(0x9E3779B97F4A7C15)


def mix64(h: Tensor) -> Tensor:
    """Finalizer of the int64 block hashes (wraparound intended)."""
    h = h ^ (h >> 29)
    h = h * 0x5BD1E995
    return h ^ (h >> 32)


def hash_bucket(h: Tensor, bits: int) -> Tensor:
    """Table row in ``[1, 2**bits]`` of a 64-bit hash (row 0 is the empty string)."""
    return torch.remainder(h, 1 << bits) + 1


def last_end_index(ends: Tensor) -> Tensor:
    """Position of the newest end at or before each byte, -1 before the first."""
    t = torch.arange(ends.size(1), device=ends.device)
    return torch.where(ends, t, -1).cummax(1).values


def block_prefix_ids(x: Tensor, ends: Tensor, bits: int, powers: Tensor) -> Tensor:
    """`hash_bucket` of `block_prefix_hash`."""
    return hash_bucket(block_prefix_hash(x, ends, powers), bits)


def block_ngram_hash(block_hash: Tensor, order: int) -> Tensor:
    """64-bit hash of the ``order`` blocks ending at each trunk slot.

    ``block_hash`` is ``(B, S)``, one completed block per slot in order;
    slots before the row start count as an empty block (hash 0). Causal: slot
    ``k`` reads slots ``k - order + 1 .. k``.
    """
    h = block_hash
    for lag in range(1, order):
        h = mix64(h * NGRAM_MULTIPLIER + F.pad(block_hash, (lag, 0))[:, : block_hash.size(1)])
    return h


def block_context_hash(prefix_hash: Tensor, ends: Tensor) -> Tensor:
    """64-bit hash of (previous completed block, current prefix) at each byte.

    At an end the current prefix is empty, so the hash is of the block just
    completed alone; before the first end the previous block is empty.
    """
    last = last_end_index(ends)
    previous = torch.where(last >= 0, torch.gather(prefix_hash, 1, last.clamp(min=0)), 0)
    current = torch.where(ends, 0, prefix_hash)
    return mix64(current * NGRAM_MULTIPLIER + previous)


def block_prefix_hash(x: Tensor, ends: Tensor, powers: Tensor) -> Tensor:
    """64-bit hash of the block string so far at each byte.

    At byte ``t`` the string is ``x[s .. t]``, where ``s`` follows the last
    patch end strictly before ``t``: the completed block's string at an end,
    otherwise the current block's prefix. The polynomial hash
    ``sum_i (x_i + 1) P**(t - i)`` mod 2**64 is a difference of prefix sums
    of ``(x_i + 1) P**-i``, so any block length costs O(1) passes, and equal
    strings get equal hashes wherever they sit (int64 wraparound is
    intended). Causal: it reads bytes and ends ``<= t``.
    """
    T = x.size(1)
    pw, ipw = powers[0, :T], powers[1, :T]
    prefix = torch.cumsum((x.long() + 1) * ipw, dim=1)
    last_end = last_end_index(ends)
    before = torch.cat([torch.full_like(last_end[:, :1], -1), last_end[:, :-1]], dim=1)
    base = torch.where(before >= 0, torch.gather(prefix, 1, before.clamp(min=0)), 0)
    return mix64((prefix - base) * pw)


def budgeted_ends(natural: Tensor, cap: int, budget: int) -> Tensor:
    """Patch ends for a trunk of ``budget`` slots per row.

    Ends (natural, and the cap's forced ones) are kept in order while the
    count so far leaves room for the forced ends the rest of the row may
    still need, ``ceil((T - t) / cap)``. From the first end that does not
    fit, natural ends are dropped and only the cap ends patches. Whether an
    end at ``t`` survives depends only on ends ``<= t`` (causal). A row
    never exceeds ``budget`` patches nor a patch ``cap`` bytes, and the
    reserve shrinks along the row instead of holding ``T // cap`` slots.

    Bound: if the last kept end is rank ``R`` at ``t' < t*`` (the first
    end that did not fit), ``R <= budget - ceil((T - t') / cap)``, and at
    most ``ceil((T - t*) / cap) <= ceil((T - t') / cap)`` forced ends
    follow.
    """
    T = natural.size(1)
    ends = capped_ends(natural, cap)
    room = budget - (T - torch.arange(T, device=natural.device) + cap - 1) // cap
    fits = ~ends | (ends.long().cumsum(1) <= room)
    return capped_ends(natural & ((~fits).cumsum(1) == 0), cap)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class TokenLM(nn.Module):
    """The GPT-2 control: embedding -> trunk -> untied softcapped head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.trunk.dim
        # bf16 storage, as in the token reference (fused AdamW keeps bf16 state).
        self.embed = nn.Embedding(GPT2_VOCAB, d).bfloat16()
        self.in_norm = RMSNorm(d)
        self.trunk = Trunk(config.trunk)
        self.out_norm = RMSNorm(d)
        self.head = Linear(d, GPT2_VOCAB)

    def prepare(self, device, lengths: tuple[int, ...] = ()) -> None:
        pass

    def forward(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, None]:
        """(per-position NLL in nats ``(B, T)`` float32, no auxiliary loss);
        ignored targets score 0."""
        x = self.trunk(self.in_norm(self.embed(inputs)))
        logits = softcap(self.head(self.out_norm(x)).float())
        nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none", ignore_index=-100)
        return nll.view_as(targets), None


class ByteEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        local, hashing = config.local, config.hashing
        self.hashing = hashing
        # bf16 storage, like the token reference's embedding table.
        self.byte_embed = nn.Embedding(BYTE_VOCAB, local.dim).bfloat16()
        self.ngram_embed = nn.Embedding(len(hashing.orders) * hashing.buckets, local.dim).bfloat16()
        self.in_norm = RMSNorm(local.dim)
        self.blocks = nn.ModuleList(LocalBlock(local, mixer) for mixer in local.mixers("encoder"))

    def forward(self, x: Tensor, block_mask: BlockMask) -> Tensor:
        h = self.byte_embed(x).float()
        with torch.no_grad():
            ids = hashed_ngram_ids(x, self.hashing.orders, self.hashing.buckets)
        # A running sum lets Inductor fuse the gathers instead of
        # materializing one (B, T, d) float32 slab per order.
        for order_ids in ids.unbind(0):
            h = h + self.ngram_embed(order_ids).float()
        h = self.in_norm(h).bfloat16()
        for block in self.blocks:
            h = block(h, block_mask)
        return h


class ByteDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        local = config.local
        dg = config.trunk.dim
        self.bos = nn.Parameter(torch.zeros(dg))
        self.route_norm = RMSNorm(dg)
        # One conditioning projection per decoder block; block 0's is live at
        # init, later ones start at zero (standard residual-branch init).
        self.inject = nn.ModuleList(Linear(dg, local.dim, bias=False) for _ in range(local.decoder_layers))
        self.blocks = nn.ModuleList(LocalBlock(local, mixer) for mixer in local.mixers("decoder"))
        self.out_norm = RMSNorm(local.dim)
        self.head = Linear(local.dim, BYTE_VOCAB)
        self.trunk_head = Linear(dg, 4 * BYTE_VOCAB, bias=False) if config.trunk_logits else None
        self.prefix_embed = self.prefix_gain = self.prefix_out = self.prefix_head = None
        if config.prefix_bits:
            rows = (1 << config.prefix_bits) + 1
            # bf16 storage, like the other lookup tables.
            self.prefix_embed = nn.Embedding(rows, local.dim).bfloat16()
            self.prefix_gain = nn.Parameter(torch.zeros(local.dim))
            self.prefix_out = nn.Embedding(rows, dg).bfloat16()
            self.prefix_head = Linear(dg, BYTE_VOCAB, bias=False)
        self.prefix_local = Linear(local.dim, dg, bias=False) if config.prefix_local else None
        self.context_embed = self.context_gain = self.context_out = None
        self.memory_gate = config.memory_gate
        if config.prefix_context:
            self.context_embed = nn.Embedding(rows, local.dim).bfloat16()
            self.context_gain = nn.Parameter(torch.zeros(local.dim))
            # Zero-init and added to `prefix_out`: the modulation starts unchanged.
            self.context_out = nn.Embedding(rows, dg).bfloat16()

    def forward(
        self,
        h_enc: Tensor,
        trunk_out: Tensor,
        route_index: Tensor,
        targets: Tensor,
        block_mask: BlockMask,
        confidence: Tensor | None = None,
        offset: Tensor | None = None,
        prefix_id: Tensor | None = None,
        context_id: Tensor | None = None,
    ) -> Tensor:
        B = h_enc.size(0)
        g = torch.cat([self.bos.type_as(trunk_out).expand(B, 1, -1), trunk_out], dim=1)
        g = self.route_norm(torch.gather(g, 1, route_index.unsqueeze(-1).expand(-1, -1, g.size(-1))))
        if confidence is not None:  # straight-through term, forward value 0
            g = g + confidence.unsqueeze(-1).type_as(g) * g
        x = h_enc
        if self.prefix_embed is not None:
            x = x + memory_read(self.prefix_gain, self.prefix_embed(prefix_id), h_enc, self.memory_gate)
        if self.context_embed is not None:
            x = x + memory_read(self.context_gain, self.context_embed(context_id), h_enc, self.memory_gate)
        for inject, block in zip(self.inject, self.blocks):
            x = block(x + inject(g), block_mask)
        x = self.out_norm(x)
        logits = self.head(x)
        if self.prefix_head is not None:
            # A readout of the trunk state chosen by the block prefix so far.
            modulation = self.prefix_out(prefix_id)
            if self.context_out is not None:
                modulation = modulation + self.context_out(context_id)
            readout = g if self.prefix_local is None else g + self.prefix_local(x).type_as(g)
            logits = logits + self.prefix_head(readout * modulation.type_as(g))
        if self.trunk_head is not None:
            by_offset = self.trunk_head(g).unflatten(-1, (4, BYTE_VOCAB))
            bucket = offset.clamp(max=3).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, BYTE_VOCAB)
            logits = logits + torch.gather(by_offset, 2, bucket).squeeze(2)
        logits = softcap(logits.float())
        return F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="none", ignore_index=-100
        ).view_as(targets)


class NextLatDecoder(nn.Module):
    """NextLat-style recurrent unembedder (see module docstring).

    The recurrence runs level by level over offsets within a patch, and each
    level touches only the patches still open at that offset: patches are
    sorted once by length, so level ``j`` is the first ``n_j`` of them. That
    is one MLP step per byte that is not a patch end, with no padding to the
    length cap; sizing the levels costs one host sync per call. ``step`` is
    compiled with dynamic shapes because the level sizes are data-dependent.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d, dg = config.local.dim, config.trunk.dim
        self.cap = config.patch_cap
        hidden = 128 * max(1, round(config.nextlat_proj_factor * 2 * d / 128))
        self.bos = nn.Parameter(torch.zeros(dg))
        self.route_norm = RMSNorm(dg)
        self.lift = Linear(dg, d, bias=False)
        self.in_norm = RMSNorm(2 * d)
        self.fc1 = Linear(2 * d, hidden, bias=False)
        self.fc2 = Linear(hidden, hidden, bias=False)
        self.fc3 = Linear(hidden, d, bias=False)
        self.out_norm = RMSNorm(d)
        self.head = Linear(d, BYTE_VOCAB)
        # Optional: the reset at a patch end also sees the encoder state of the
        # end byte directly, not only through the trunk (zero-init: off at start).
        self.reset = Linear(d, d, bias=False) if config.nextlat_local_reset else None
        self.compiled: dict[str, object] = {}

    def compile(self) -> None:
        self.compiled["step"] = torch.compile(self.step, dynamic=True)
        self.compiled["readout"] = torch.compile(self.readout, dynamic=False)

    def start(self, g: Tensor) -> Tensor:
        return self.lift(self.route_norm(g)).float()

    def step(self, state: Tensor, action: Tensor) -> Tensor:
        # The residual state stays fp32 across the unroll; the MLP runs in bf16.
        u = self.in_norm(torch.cat([action.type_as(state), state], dim=-1))
        return state + self.fc3(F.gelu(self.fc2(F.gelu(self.fc1(u))))).float()

    def readout(self, s: Tensor, targets: Tensor, confidence: Tensor | None = None, trunk_part: Tensor | None = None) -> Tensor:
        if confidence is not None:
            # Straight-through (forward value 0) on the trunk's part of the
            # state, the start state lifted from it; scaling all of ``s``
            # would be divided back out by ``out_norm``.
            s = s + confidence.unsqueeze(-1) * trunk_part
        logits = softcap(self.head(self.out_norm(s)).float())
        return F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="none", ignore_index=-100
        ).view_as(targets)

    def forward(
        self,
        h_enc: Tensor,
        trunk_out: Tensor,
        route_index: Tensor,
        offset: Tensor,
        targets: Tensor,
        confidence: Tensor | None = None,
    ) -> Tensor:
        B, T, d = h_enc.shape
        step = self.compiled.get("step", self.step)
        readout = self.compiled.get("readout", self.readout)
        g = torch.cat([self.bos.type_as(trunk_out).expand(B, 1, -1), trunk_out], dim=1)
        k1 = g.size(1)
        # Deepest offset each route reaches: route 0 (before the first end)
        # holds offsets 1..count, a real patch 0..count-1; padding never runs.
        count = torch.zeros(B, k1, dtype=torch.long, device=h_enc.device)
        count.scatter_add_(1, route_index, torch.ones_like(route_index))
        depth = (count - (torch.arange(k1, device=h_enc.device) > 0).long()).flatten()
        order = torch.argsort(depth, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(order.numel(), device=order.device)
        levels = (depth[None, :] >= torch.arange(1, self.cap, device=depth.device)[:, None]).sum(1)
        sizes = [n for n in levels.tolist() if n]  # the one host sync (non-increasing)
        # Byte consumed by (route, level j) sits at base + j: base is the
        # route's patch-end byte, or -1 for the BOS route.
        rows = torch.arange(B, device=h_enc.device)[:, None]
        t = torch.arange(T, device=h_enc.device).expand(B, -1)
        # Patch-end bytes (offset 0) scatter into their route; the rest into
        # a discarded extra column. Route 0 has no end byte and keeps -1.
        base = torch.full((B, k1 + 1), -1, dtype=torch.long, device=h_enc.device)
        base = base.scatter(1, torch.where(offset == 0, route_index, k1), t)[:, :k1]
        base_flat = (base + rows * T).flatten()[order]
        h_flat = h_enc.flatten(0, 1)
        lifted = self.start(g).flatten(0, 1)[order]  # the trunk's part of each start state
        s0 = lifted
        if self.reset is not None:
            # The BOS route has no end byte (base -1): no local term.
            local = h_flat[base_flat.clamp(min=0)] * (base_flat >= 0).unsqueeze(-1)
            s0 = s0 + self.reset(local).float()
        states = [s0]
        for j, n in enumerate(sizes, start=1):
            states.append(step(states[-1][:n], h_flat[base_flat[:n] + j]))
        starts = torch.tensor([0, *[x.size(0) for x in states]], device=h_enc.device).cumsum(0)
        flat = torch.cat(states)
        route = rank[(route_index + rows * k1)]
        trunk_part = lifted[route] if confidence is not None else None
        return readout(flat[starts[offset] + route], targets, confidence, trunk_part)


class ThresholdGate(nn.Module):
    """A per-byte gate score whose threshold decides natural patch ends.

    ``threshold`` is a buffer moved only by `update_threshold`, which a
    training forward calls after routing, so a batch's boundaries use only
    past batches and stay a deterministic function of bytes ``<= t``.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.register_buffer("threshold", torch.tensor(float("nan")))

    def natural_ends(self, gate: Tensor) -> Tensor:
        return gate.detach() > self.threshold  # NaN threshold: no natural ends

    @torch.no_grad()
    def update_threshold(self, gate: Tensor, valid: Tensor, budget: int, momentum: float = 0.9) -> None:
        """EMA toward this batch's threshold (used from the next batch on).

        ``rate``: the threshold whose *realized* patch rate on this batch --
        natural ends within the chunk budget plus the cap's forced ends -- is
        ``1 / target_patch``, found by bisection (the rate falls monotonically
        as it rises), so a rate-L run spends the same trunk compute per byte
        as static-L whatever the gate learns.
        ``mean``: the mean gate score; for entropy, a patch continues while
        the next byte is predicted to be easier than average.
        ``fixed``: unchanged.
        """
        mode = self.config.threshold_mode
        if mode == "fixed":
            return
        e = gate.detach().float()
        if mode == "rate":
            lo = torch.where(valid, e, torch.inf).amin() - 1
            hi = torch.where(valid, e, -torch.inf).amax()
            n = valid.sum()
            for _ in range(24):
                mid = (lo + hi) / 2
                rate = (budgeted_ends(e > mid, self.config.max_patch, budget) & valid).sum() / n
                above = rate > 1 / self.config.target_patch
                lo, hi = torch.where(above, mid, lo), torch.where(above, hi, mid)
            value = hi
        else:
            value = torch.where(valid, e, torch.nan).nanmean()
        self.threshold.copy_(torch.where(self.threshold.isnan(), value, self.threshold.lerp(value, 1 - momentum)))


class EntropyPatcher(ThresholdGate):
    """Local next-byte head whose entropy decides where patches end."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.norm = RMSNorm(config.local.dim)
        self.head = Linear(config.local.dim, BYTE_VOCAB)

    def forward(self, h: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """(entropy of byte t+1 in nats, local CE of byte t+1)."""
        if self.config.aux_detach:
            h = h.detach()
        logits = softcap(self.head(self.norm(h)).float())
        logp = logits.log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        nll = F.nll_loss(logp.flatten(0, 1), targets.flatten(), reduction="none", ignore_index=-100)
        return entropy, nll.view_as(targets)


class PatchPool(nn.Module):
    """Multi-head attention pooling of each patch's bytes with learned queries.

    Patch ``k`` is bytes ``(end_{k-1}, end_k]``, and each patch pools only its
    own bytes (a segment softmax), so a patch's trunk input still depends
    only on bytes up to its end. The output is added to the end-byte state
    through a zero-initialized value projection, so a fresh model starts as
    end-state pooling.

    Segment sums are differences of float64 prefix sums taken at the patch
    ends: deterministic (scatter-add atomics are not, and their run-to-run
    rounding breaks exact causality checks) and accurate to ~T ulps of 2^-52.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.local.dim
        self.heads = config.pool_heads
        self.head_dim = d // self.heads
        self.norm = RMSNorm(d)
        self.key = Linear(d, d, bias=False)
        self.value = Linear(d, d, bias=False)
        self.query = nn.Parameter(torch.zeros(d))

    def forward(self, h: Tensor, patch_id: Tensor, end_index: Tensor, ends: Tensor) -> Tensor:
        """h (B, T, d), patch_id (B, T) in [0, K], end_index (B, K), ends (B, K, d) -> (B, K, d)."""
        B, T, d = h.shape
        K = ends.size(1)
        x = self.norm(h)
        k = self.key(x).view(B, T, self.heads, self.head_dim)
        v = self.value(x).view(B, T, self.heads, self.head_dim)
        q = self.query.view(self.heads, self.head_dim).type_as(k)
        score = (k * q).sum(-1).float() / self.head_dim**0.5  # (B, T, H)
        # Per-patch max as a softmax shift (order-independent, so exact; the
        # softmax is shift-invariant, so it carries no gradient). Bytes after
        # the last end land in an extra slot.
        pid = patch_id.unsqueeze(-1).expand(-1, -1, self.heads)
        peak = score.detach().new_full((B, K + 1, self.heads), -torch.inf).scatter_reduce(1, pid, score.detach(), "amax")
        w = (score - peak.gather(1, pid)).exp()  # (B, T, H), 1 at each patch's peak
        prev = torch.cat([end_index.new_full((B, 1), -1), end_index[:, :-1]], 1)
        real = end_index > prev  # padding patches point at byte 0

        def segment_sum(values: Tensor) -> Tensor:
            """(B, T, ...) -> (B, K, ...): sum over each patch's bytes."""
            c = F.pad(values.double().cumsum(1), (0, 0) * (values.ndim - 2) + (1, 0))  # c[t+1] = sum_{<=t}
            shape = (B, K) + values.shape[2:]
            at = lambda i: c.gather(1, i.view(B, K, *([1] * (values.ndim - 2))).expand(shape))
            return at(end_index + 1) - at(prev + 1)

        denom = segment_sum(w).clamp(min=1)  # >= 1 on real patches (the peak term)
        pooled = segment_sum(w.unsqueeze(-1) * v.float()) / denom.unsqueeze(-1)
        pooled = torch.where(real[:, :, None, None], pooled, 0).to(ends.dtype)
        return ends + pooled.reshape(B, K, d)


class LearnedRouter(ThresholdGate):
    """Causal continue/stop gate trained through the main loss (see module
    docstring). Its threshold is 0.5 (``fixed``, as in H-Net) or held at the
    target rate (``rate``)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        d = config.local.dim
        self.norm = RMSNorm(d)
        self.query = Linear(d, d, bias=False)
        self.key = Linear(d, d, bias=False)

    def forward(self, h: Tensor) -> Tensor:
        """p_t, the probability of ending a chunk after byte t, (B, T) float32."""
        x = self.norm(h)
        q, k = self.query(x).float(), self.key(x).float()
        previous = F.pad(k[:, :-1], (0, 0, 1, 0))  # zero before byte 0: p_0 = 1/2
        return ((1 - F.cosine_similarity(q, previous, dim=-1)) / 2).clamp(0, 1)


def ratio_loss(p: Tensor, valid: Tensor, target: float) -> Tensor:
    """H-Net's rate loss: minimal (= 1) when the hard end rate F and the mean
    probability G are both ``1 / target``; the gradient flows through G.

    F counts ``p > 0.5``, the router's own decisions, not the realized ends:
    the rate controller holds the realized rate at the target, where the loss
    is constant in G, and p then drifted to saturation (NOTES 2026-09-30)."""
    n = valid.sum()
    f = (p.detach() > 0.5).logical_and(valid).sum() / n
    g = torch.where(valid, p, 0).sum() / n
    return target / (target - 1) * ((target - 1) * f * g + (1 - f) * (1 - g))


def smooth_chunks(z: Tensor, weight: Tensor, steps: int) -> Tensor:
    """``out_k = w_k z_k + (1 - w_k) out_{k-1}`` over dim 1 (``out_{-1} = 0``)
    by a Hillis-Steele scan of the affine maps: exact, deterministic, and
    ``steps`` fixed (``2**steps >= K``) so a dynamic K never recompiles."""
    a = (1 - weight).unsqueeze(-1)
    b = weight.unsqueeze(-1) * z.float()
    for i in range(steps):
        s = 1 << i
        b = torch.cat([b[:, :s], b[:, s:] + a[:, s:] * b[:, :-s]], 1)
        a = torch.cat([a[:, :s], a[:, s:] * a[:, :-s]], 1)
    return b.type_as(z)


class ByteLM(nn.Module):
    """Hierarchical byte model around the KDA8 trunk."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = ByteEncoder(config)
        self.patcher = {"entropy": EntropyPatcher, "learned": LearnedRouter}.get(config.patcher, lambda c: None)(config)
        # Weight of the patcher's auxiliary loss in the training objective.
        self.aux_scale = config.aux_weight if config.patcher == "entropy" else config.ratio_weight
        self.pool = PatchPool(config) if config.patch_pool == "attn" else None
        self.patch_proj = Linear(config.local.dim, config.trunk.dim, bias=False)
        self.block_embed = self.block_gain = None
        if config.prefix_bits:
            self.block_embed = nn.Embedding((1 << config.prefix_bits) + 1, config.trunk.dim).bfloat16()
            self.block_gain = nn.Parameter(torch.zeros(config.trunk.dim))
        # Orders 2 .. block_ngram, one table and gain each.
        self.block_ngram_embed = nn.ModuleList(
            nn.Embedding((1 << config.prefix_bits) + 1, config.trunk.dim).bfloat16()
            for _ in range(config.block_ngram - 1)
        )
        self.block_ngram_gain = nn.ParameterList(
            nn.Parameter(torch.zeros(config.trunk.dim)) for _ in range(config.block_ngram - 1)
        )
        self.prefix_powers: dict[int, Tensor] = {}
        self.patch_norm = RMSNorm(config.trunk.dim)
        self.trunk = Trunk(config.trunk)
        self.trunk_norm = RMSNorm(config.trunk.dim)
        self.decoder = NextLatDecoder(config) if config.decoder == "nextlat" else ByteDecoder(config)
        self.block_masks: dict[int, BlockMask] = {}  # the decoder's
        self.encoder_masks: dict[int, BlockMask] = {}
        # Compiled callables live in a plain dict so they are not registered
        # as submodules (which would duplicate parameters in state_dict).
        self.compiled: dict[str, object] = {}
        self.last_layout: PatchLayout | None = None
        self.last_stats: dict[str, Tensor] = {}
        # Gate score of the last forward: entropy in nats, or the router's p.
        self.last_gate: Tensor | None = None
        self.last_trunk_args: tuple = ()  # the trunk's extra inputs (component timing)
        self.last_confidence: Tensor | None = None

    def compile_parts(self) -> None:
        """Compile encoder, patcher, trunk and decoder separately, all with
        static shapes: one graph per row length, since the trunk length is
        the fixed chunk budget (`ModelConfig.chunk_budget`)."""
        self.compiled["encoder"] = torch.compile(self.encoder, dynamic=False)
        self.compiled["trunk"] = torch.compile(self.run_trunk, dynamic=False)
        if isinstance(self.decoder, NextLatDecoder):
            self.decoder.compile()  # compiles its own step and readout
        else:
            self.compiled["decoder"] = torch.compile(self.decoder, dynamic=False)
        if self.patcher is not None:
            self.compiled["patcher"] = torch.compile(self.patcher, dynamic=False)

    def part(self, name: str):
        if name in self.compiled:
            return self.compiled[name]
        return {"encoder": self.encoder, "trunk": self.run_trunk, "decoder": self.decoder, "patcher": self.patcher}[name]

    def run_trunk(
        self,
        ends: Tensor,
        h: Tensor | None = None,
        patch_id: Tensor | None = None,
        end_index: Tensor | None = None,
        weight: Tensor | None = None,
        block_ids: Tensor | None = None,
    ) -> Tensor:
        """``block_ids`` ``(B, S, block_ngram)``: per slot, the completed
        block's id, then the ids of the n-grams of blocks ending there."""
        if self.pool is not None:
            ends = self.pool(h, patch_id, end_index, ends)
        x = state = self.patch_proj(ends)
        if block_ids is not None:
            gains = [self.block_gain, *self.block_ngram_gain]
            tables = [self.block_embed, *self.block_ngram_embed]
            for gain, table, ids in zip(gains, tables, block_ids.unbind(-1)):
                x = x + memory_read(gain, table(ids), state, self.config.memory_gate)
        z = self.trunk_norm(self.trunk(self.patch_norm(x)))
        if weight is not None:  # learned router: p at each chunk end
            z = smooth_chunks(z, weight, steps=math.ceil(math.log2(z.size(1))))
        return z

    def prepare(self, device, lengths: tuple[int, ...] = ()) -> None:
        """Build sliding-window masks outside any compiled region, for the
        training row length and any evaluation row lengths."""
        for length in {self.config.seq_len, *lengths}:
            if length % self.config.patch_size:
                raise ValueError(f"row length {length} is not a multiple of the patch size")
            local = self.config.local
            self.block_masks[length] = sliding_window_mask(length, local.window, device)
            self.encoder_masks[length] = (
                self.block_masks[length]
                if local.stage_window("encoder") == local.window
                else sliding_window_mask(length, local.stage_window("encoder"), device)
            )
            if self.config.prefix_bits:
                self.prefix_powers[length] = prefix_powers(length, device)

    def forward(
        self,
        inputs: Tensor,
        targets: Tensor,
        ablate_global: bool = False,
        boundaries: Callable[[Tensor], Tensor] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """(per-position NLL in nats ``(B, T)``, auxiliary local NLL or None).

        ``ablate_global`` routes every byte to the BOS state (a knock-out the
        decoder never trained with; it bounds what the trunk contributes).
        ``boundaries`` (diagnostic, gated patchers) replaces the gate's
        natural ends: it maps the gate score ``(B, T)`` to causal ends, e.g.
        random or static ends at the gate's rate, to test where the gate
        puts them (the cap and chunk budget still apply).
        """
        B, T = inputs.shape
        block_mask = self.block_masks[T]
        h = self.part("encoder")(inputs, self.encoder_masks[T])
        aux = None
        confidence = weight = None
        if self.config.patcher == "static":
            layout = PatchLayout.static(B, T, self.config.patch_size, inputs.device)
        elif self.config.patcher == "space":
            budget = self.config.chunk_budget(T)
            natural = spacelike_ends(inputs)
            ends = budgeted_ends(natural, self.config.max_patch, budget)
            layout = PatchLayout.from_ends(ends, budget)
            self.last_stats = {
                "patches_per_byte": layout.patch_valid.sum() / (B * T),
                "forced_end_fraction": (ends & ~natural).sum() / ends.sum(),
                "dropped_end_fraction": (natural & ~ends).sum() / natural.sum().clamp(min=1),
            }
        else:
            valid = targets >= 0
            if self.config.patcher == "entropy":
                gate, aux = self.part("patcher")(h, targets)
            else:
                gate = self.part("patcher")(h)
            budget = self.config.chunk_budget(T)
            natural = self.patcher.natural_ends(gate) if boundaries is None else boundaries(gate)
            ends = budgeted_ends(natural, self.config.max_patch, budget)
            layout = PatchLayout.from_ends(ends, budget)
            if self.training and torch.is_grad_enabled():
                # After routing: this batch's boundaries used only past batches.
                self.patcher.update_threshold(gate, valid, budget)
            stats = {}
            if self.config.patcher == "entropy":
                stats["local_entropy"] = gate.detach().mean()
            else:
                rate = ratio_loss(gate, valid, self.config.target_patch)
                aux = rate * valid  # sums to n_valid * loss, like the summed CE
                c = torch.where(ends, gate, 1 - gate)
                confidence = c - c.detach()  # straight-through: forward exactly 0
                weight = torch.gather(gate, 1, layout.end_index)
                stats |= {
                    "ratio_loss": rate.detach(),
                    "mean_p": gate.detach().mean(),
                    "router_end_rate": (gate.detach() > 0.5)[valid].float().mean(),
                }
            self.last_stats = {
                "threshold": self.patcher.threshold.detach().clone(),
                "patches_per_byte": layout.patch_valid.sum() / (B * T),
                "forced_end_fraction": (ends & ~natural).sum() / ends.sum(),
                "dropped_end_fraction": (natural & ~ends).sum() / natural.sum().clamp(min=1),
                **stats,
            }
            self.last_gate = gate.detach()
        ends = self.gather_ends(h, layout)
        pool_args: tuple = (None, None, None)
        if self.pool is not None:
            # Patch of byte t: the number of patch ends strictly before t.
            patch_id = layout.route_index - (layout.offset == 0).long()
            pool_args = (h, patch_id, layout.end_index)
        block_ids = None
        if bits := self.config.prefix_bits:
            is_end = layout.offset == 0
            prefix_hash = block_prefix_hash(inputs, is_end, self.prefix_powers[T])
            block_hash = torch.gather(prefix_hash, 1, layout.end_index)  # each completed block's string
            block_ids = torch.stack(
                [hash_bucket(block_ngram_hash(block_hash, n), bits) for n in range(1, self.config.block_ngram + 1)],
                dim=-1,
            )
            # The decoder reads the current block's prefix: empty (id 0) after an end.
            layout = replace(layout, prefix_id=torch.where(is_end, 0, hash_bucket(prefix_hash, bits)))
            if self.config.prefix_context:
                layout = replace(layout, context_id=hash_bucket(block_context_hash(prefix_hash, is_end), bits))
        trunk_args = (*pool_args, weight, block_ids)
        z = self.part("trunk")(ends, *trunk_args)
        self.last_layout = layout
        self.last_trunk_args = trunk_args
        self.last_confidence = confidence
        return self.decode(h, z, layout, targets, block_mask, ablate_global, confidence), aux

    @staticmethod
    def gather_ends(h: Tensor, layout: PatchLayout) -> Tensor:
        return torch.gather(h, 1, layout.end_index.unsqueeze(-1).expand(-1, -1, h.size(-1)))

    def decode(self, h, z, layout: PatchLayout, targets, block_mask, ablate_global=False, confidence=None) -> Tensor:
        if ablate_global:
            # Replace every patch state by BOS but keep the real routes, so
            # the recurrent decoder's grid slots stay unique (and causal).
            z = self.decoder.bos.detach().type_as(z).expand_as(z).contiguous()
        if self.config.decoder == "nextlat":
            return self.part("decoder")(h, z, layout.route_index, layout.offset, targets, confidence)
        return self.part("decoder")(
            h, z, layout.route_index, targets, block_mask, confidence, layout.offset, layout.prefix_id, layout.context_id
        )


def build_model(config: ModelConfig) -> nn.Module:
    return TokenLM(config) if config.atom == "gpt2" else ByteLM(config)


# --------------------------------------------------------------------------
# Initialization (mirrors `initialize_model` in the token trainer)
# --------------------------------------------------------------------------


def _normal(weight: Tensor) -> None:
    weight.normal_(std=0.33**0.5 / weight.size(-1) ** 0.5)


def _zero(linear: nn.Linear) -> None:
    linear.weight.zero_()
    if linear.bias is not None:
        linear.bias.zero_()


@torch.no_grad()
def initialize(model: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    for p in model.parameters():
        p.fill_(float("nan"))
    for module in model.modules():
        if isinstance(module, KimiDeltaAttention):
            for linear in (
                module.q_proj,
                module.k_proj,
                module.v_proj,
                module.f_a_proj,
                module.f_b_proj,
                module.b_proj,
                module.g_a_proj,
                module.g_b_proj,
            ):
                _normal(linear.weight)
            # Identity causal convolution preserves the current-token projection.
            for conv in (module.q_conv1d, module.k_conv1d, module.v_conv1d):
                conv.weight.zero_()
                conv.weight[:, 0, -1] = 1
            module.A_log.zero_()
            log_dt = torch.empty_like(module.dt_bias).uniform_(math.log(0.001), math.log(0.1))
            dt = log_dt.exp().clamp(min=1e-4)
            module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            module.o_norm.weight.fill_(1)
            _zero(module.o_proj)
        elif isinstance(module, (CausalSelfAttention, LocalAttention)):
            for linear in (module.q, module.k, module.v):
                _normal(linear.weight)
                linear.bias.zero_()
            _zero(module.proj)
        elif isinstance(module, ReluSquaredMLP):
            _normal(module.fc.weight)
            module.fc.bias.zero_()
            _zero(module.proj)
        elif isinstance(module, RMSNorm):
            module.gains.fill_(1)
    if isinstance(model, TokenLM):
        model.embed.weight.normal_()
        _zero(model.head)
    else:
        model.encoder.byte_embed.weight.normal_()
        model.encoder.ngram_embed.weight.normal_()
        _normal(model.patch_proj.weight)
        decoder = model.decoder
        decoder.bos.zero_()
        if isinstance(decoder, NextLatDecoder):
            for linear in (decoder.lift, decoder.fc1, decoder.fc2):
                _normal(linear.weight)
            _zero(decoder.fc3)  # the recurrence starts as the identity
        else:
            _normal(decoder.inject[0].weight)
            for inject in decoder.inject[1:]:
                _zero(inject)
        _zero(decoder.head)
        if getattr(decoder, "trunk_head", None) is not None:
            _zero(decoder.trunk_head)
        if model.block_embed is not None:
            model.block_gain.zero_()
            decoder.prefix_gain.zero_()
            _zero(decoder.prefix_head)
            # The modulator starts at one: an unseen prefix reads the plain
            # trunk readout instead of a random one.
            decoder.prefix_out.weight.fill_(1)
            for gain in model.block_ngram_gain:
                gain.zero_()
            if decoder.prefix_local is not None:
                _zero(decoder.prefix_local)
            if decoder.context_out is not None:
                decoder.context_gain.zero_()
                decoder.context_out.weight.zero_()
        if isinstance(decoder, NextLatDecoder) and decoder.reset is not None:
            _zero(decoder.reset)
        if model.pool is not None:
            _normal(model.pool.key.weight)
            _zero(model.pool.value)
            model.pool.query.normal_()
        if isinstance(model.patcher, EntropyPatcher):
            _zero(model.patcher.head)
        elif isinstance(model.patcher, LearnedRouter):
            # Identity projections (as in H-Net): p starts as the cosine
            # distance between consecutive byte states.
            for linear in (model.patcher.query, model.patcher.key):
                linear.weight.copy_(torch.eye(linear.weight.size(0)))
        if model.patcher is not None:
            # Rate and mean thresholds are set by the first training batch.
            fixed = model.config.threshold_mode == "fixed"
            model.patcher.threshold.fill_(0.5 if fixed else float("nan"))
        if model.block_embed is not None:
            # Last, so every other draw matches the model without the memory.
            model.block_embed.weight.normal_()
            model.decoder.prefix_embed.weight.normal_()
            # And after those, so adding n-gram or context memory leaves the
            # unigram tables' draws unchanged.
            for table in model.block_ngram_embed:
                table.weight.normal_()
            if model.decoder.context_embed is not None:
                model.decoder.context_embed.weight.normal_()
    bad = [name for name, p in model.named_parameters() if not p.isfinite().all()]
    if bad:
        raise RuntimeError(f"uninitialized parameters: {bad}")


def lookup_tables(model: "ByteLM") -> list[nn.Embedding]:
    """A byte model's embedding tables: gathered, never multiplied."""
    tables = [model.encoder.byte_embed, model.encoder.ngram_embed]
    if model.block_embed is not None:
        tables += [model.block_embed, *model.block_ngram_embed, model.decoder.prefix_embed, model.decoder.prefix_out]
    if getattr(model.decoder, "context_embed", None) is not None:
        tables += [model.decoder.context_embed, model.decoder.context_out]
    return tables


def parameter_report(model: nn.Module) -> dict[str, int]:
    """Parameter counts split into lookup tables and compute-bearing weights."""
    lookup_ids = set()
    if isinstance(model, TokenLM):
        lookup_ids.add(id(model.embed.weight))
    else:
        lookup_ids |= {id(t.weight) for t in lookup_tables(model)}
    total = sum(p.numel() for p in model.parameters())
    lookup = sum(p.numel() for p in model.parameters() if id(p) in lookup_ids)
    trunk = sum(p.numel() for p in model.trunk.parameters())
    return {"total": total, "lookup": lookup, "compute": total - lookup, "trunk": trunk}


def flops_per_atom(model: nn.Module, config: ModelConfig) -> dict[str, float]:
    """Training matmul FLOPs (6 x weights touched) per atom, by component.

    Attention score FLOPs are added for softmax layers; KDA's recurrent cost is
    O(T d^2 / heads) and is folded into its projections as an approximation.
    """
    report: dict[str, float] = {}
    trunk_weights = sum(p.numel() for p in model.trunk.parameters() if p.ndim == 2)
    dense_layers = config.trunk.layers - len(config.trunk.kda_layers)
    if isinstance(model, TokenLM):
        ctx = config.seq_len
        report["trunk"] = 6 * trunk_weights + 6 * dense_layers * ctx * config.trunk.dim
        report["head"] = 6 * model.head.weight.numel()
        return report
    # Mean bytes per patch: nominal for the entropy patcher (rate mode controls
    # it; mean mode does not), so compare against the logged bytes_per_patch.
    p = config.patch_size if config.patcher == "static" else config.target_patch
    # The trunk runs every slot of the chunk budget, used or not.
    patches = config.chunk_budget(config.seq_len)
    report["trunk"] = (6 * trunk_weights + 6 * dense_layers * patches * config.trunk.dim) * patches / config.seq_len
    # Each query attends to ~window keys: QK^T and AV are 2*window*dim MACs,
    # i.e. 4*window*dim FLOPs forward, 12*window*dim for training.
    local_attn = {stage: 12 * config.local.stage_window(stage) * config.local.dim for stage in ("encoder", "decoder")}
    enc_w = sum(p_.numel() for p_ in model.encoder.blocks.parameters() if p_.ndim == 2)
    tables = {id(t.weight) for t in lookup_tables(model)}
    dec_w = sum(p_.numel() for p_ in model.decoder.parameters() if p_.ndim == 2 and id(p_) not in tables)
    report["encoder"] = 6 * enc_w + local_attn["encoder"] * config.local.mixers("encoder").count("a")
    if config.decoder == "nextlat":
        # The step MLP runs once per byte that is not a patch end, lift once
        # per patch, the head once per byte.
        dec = model.decoder
        mlp_w = sum(l.weight.numel() for l in (dec.fc1, dec.fc2, dec.fc3))
        report["decoder"] = 6 * (mlp_w * (1 - 1 / p) + dec.lift.weight.numel() / p + dec.head.weight.numel())
    else:
        report["decoder"] = 6 * dec_w + local_attn["decoder"] * config.local.mixers("decoder").count("a")
    if model.patcher is not None:
        report["patcher"] = 6 * sum(p_.numel() for p_ in model.patcher.parameters() if p_.ndim == 2)
    report["pool"] = 6 * model.patch_proj.weight.numel() / p
    if model.pool is not None:
        report["pool"] += 6 * (model.pool.key.weight.numel() + model.pool.value.weight.numel())
    return report

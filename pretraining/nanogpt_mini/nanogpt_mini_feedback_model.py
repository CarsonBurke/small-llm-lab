"""Latent feedback ("temporal residual") variants of the nanogpt-mini GPT.

The trunk is the baseline ``GPT`` (embed, blocks, head, norms; identical
registration order, so a seeded init draws the same trunk weights). What
changes is how the top-layer state of earlier positions re-enters the stack:

``mode="none"``
    The baseline. One pass, no feedback.
``mode="glu"``
    Full-bandwidth transformer (arXiv:2608.08888): the input at position t
    is ``rmsnorm(W_u s_{t-1} * sigmoid(W_g e_t))`` for t >= 1 and the plain
    ``rmsnorm(e_0)`` at t = 0, where ``s`` is the post-final-norm top state.
    The token survives only in the gate, so the state cannot be ignored.
``mode="add"``
    The additive control: ``rmsnorm(e_t + W_u s_{t-1})`` with ``W_u``
    zero-initialised, so pass 2 starts as pass 1 and the model may leave the
    channel unused (the shortcut the paper argues against).
``mode="lam"``
    Linear-attention memory over *all* earlier top states, read at every
    layer by default. ``memory_layers`` selects sorted, unique, zero-based
    read sites instead. Keys and values are projections of ``s_i``; each
    selected layer queries memory from its attention input with its own projection and
    adds the read through a zero-initialised projection. Attention weights
    are ``phi(q_t) . phi(k_i) * lambda_h^(t-i)`` for ``i < t`` with
    ``phi = elu + 1`` and a learned per-head decay, normalised by their sum,
    so the memory is a fixed-size fast-weight state
    ``(M_h, z_h) = (sum_i lambda^(t-i) phi(k_i) v_i^T, sum_i lambda^(t-i) phi(k_i))``
    at decode time and a masked matmul during parallel training.

``mode="top"``
    Single pass. The same decayed linear-attention memory over earlier top
    states, read once at the top: ``h_t = norm2(x_t)`` writes ``(k_t, v_t)``
    and queries ``r_t = read(q(h_t), k_{<t}, v_{<t})``; the head sees
    ``norm2(x_t + W_o r_t)`` with ``W_o`` zero-initialised. Because the read
    only needs earlier positions' *final* states, it is computable in the
    ordinary parallel forward, so training costs one pass (the whole trunk
    gain of the two-pass arms came through the gradient into the earlier
    top states, which this keeps). ``detach`` cuts that gradient (keys and
    values from ``h.detach()``) and leaves the read as a pure inference
    channel.

Training uses the paper's Jacobi passes: pass 1 is the ordinary forward;
pass k >= 2 rebuilds the inputs (or memory) from pass k-1's top states,
shifted so position t only sees positions < t. Every pass is scored with the
next-token loss; the total is ``loss_1 + mean_{k>=2} loss_k``. Gradients flow
through the carried states unless ``detach`` is set (the MiniCPM carry
contract). ``noise`` adds uniform jitter to the carried state in training.

``loss_sequential`` evaluates the true recurrence: tokens are processed one
at a time with a KV cache, and each position's feedback comes from the
sequentially computed top states, exactly as at decode time.

``attn_window`` restricts every layer's attention to the last ``W`` tokens
(0 = full causal, the baseline). It exists to measure how much the
temporal residual is worth when attention cannot reach the processed past
directly.
"""

from __future__ import annotations

import math
import os

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.nanogpt_mini.nanogpt_mini_model import Block, Linear, RMSNorm

MODES = ("none", "glu", "add", "lam", "top")
ATTENTION_SCALE = 0.12
DEFAULT_DECAYS = (0.5, 0.9, 0.98, 0.999)


def rotary_cos_sin(angular_freq: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
    theta = torch.outer(positions.float(), angular_freq)
    return theta.cos(), theta.sin()


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """``x`` [..., D] with cos/sin broadcastable to [..., D/2]; matches ``Rotary.forward``."""
    x1, x2 = x.float().chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1).type_as(x)


def softcap_logits(x: Tensor) -> Tensor:
    logits = x.float()
    return 15 * logits * (logits.square() + 15**2).rsqrt()


def window_mask(length: int, window: int, device: torch.device) -> Tensor:
    """[T, T] bool: query t may attend key i iff 0 <= t - i < window."""
    positions = torch.arange(length, device=device)
    lag = positions[:, None] - positions[None, :]
    return (lag >= 0) & (lag < window)


def linear_memory_features(x: Tensor, temperature: Tensor) -> Tensor:
    """``phi(temperature * rmsnorm(x))`` per head; ``x`` [..., H, d], temperature [H]."""
    normed = F.rms_norm(x.float(), (x.size(-1),)) * temperature.float()[..., :, None]
    return F.elu(normed) + 1.0


def decay_mask(decays: Tensor, length: int) -> Tensor:
    """[H, T, T] with ``lambda_h^(t-i)`` for ``i < t`` and 0 elsewhere (fp32)."""
    positions = torch.arange(length, device=decays.device, dtype=torch.float32)
    lag = positions[:, None] - positions[None, :]
    strict = lag > 0
    log_decay = decays.float().clamp_min(1e-30).log()[:, None, None]
    return torch.where(strict, (log_decay * lag.clamp_min(0)).exp(), torch.zeros_like(lag))


def linear_memory_read_dense(q: Tensor, k: Tensor, v: Tensor, decays: Tensor) -> Tensor:
    """Reference form with the full [H, T, T] decay mask (fp32); q, k, v [B, T, H, d] -> [B, T, H, d]."""
    mask = decay_mask(decays, q.size(1))
    qh, kh, vh = (x.float().transpose(1, 2) for x in (q, k, v))  # [B, H, T, d]
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * mask[None]
    numerator = torch.matmul(scores, vh)
    denominator = scores.sum(-1, keepdim=True)
    return (numerator / (denominator + 1e-6)).transpose(1, 2)


MEMORY_CHUNK = 128
MEMORY_KERNEL = os.environ.get("FB_MEMORY_KERNEL", "1") != "0"
MEMORY_COMPILED = os.environ.get("FB_MEMORY_COMPILED", "1") != "0"

try:
    from fla.ops.simple_gla import chunk_simple_gla
except ImportError:  # pragma: no cover - the kernel is optional for the CPU reference path
    chunk_simple_gla = None


def linear_memory_read_chunked(q: Tensor, k: Tensor, v: Tensor, decays: Tensor, chunk: int = MEMORY_CHUNK) -> Tensor:
    """Strictly causal decayed linear attention, chunkwise in fp32: q [B,T,H,d] over (k, v) [B,T,H,d], decays [H].

    Position ``t = cC + j`` reads ``sum_{i<t} lambda^(t-i) phi(q_t).phi(k_i) v_i`` normalised by the
    same weights: keys inside the chunk through a [C, C] decay mask, keys of earlier chunks through
    the fast-weight state ``S_c = sum_{i<cC} lambda^(cC-i) phi(k_i) v_i^T`` (and ``z_c`` for the
    normaliser) carried across chunks with ``S_{c+1} = lambda^C S_c + sum_j lambda^(C-j) k_{cC+j} v^T``.
    Everything is fp32 (the decode-time state is fp32); memory is O(T C) instead of O(T^2).
    Precondition: ``q`` and ``k`` are ``linear_memory_features`` outputs (strictly positive), so the
    normaliser is a sum of non-negative weights and ``+1e-6`` is a floor rather than a guard against
    a sign change. ``MemoryState`` carries the state one decay step behind this convention, so the two
    differ by ``1e-6 * (1 - lambda)`` in the normaliser: immaterial at ``d = 128``.
    """
    B, T, H, d = q.shape
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    if T == 0:
        return q.float().new_zeros(B, T, H, d)
    C = min(chunk, T)
    pad = (-T) % C
    qh, kh, vh = (F.pad(x.float().transpose(1, 2), (0, 0, 0, pad)) for x in (q, k, v))  # [B, H, Tp, d]
    n = (T + pad) // C
    qc, kc, vc = (x.reshape(B, H, n, C, d) for x in (qh, kh, vh))
    log_decay = decays.float().clamp_min(1e-30).log()  # [H]
    j = torch.arange(C, device=q.device, dtype=torch.float32)
    lag = j[:, None] - j[None, :]
    intra = torch.where(lag > 0, (log_decay[:, None, None] * lag.clamp_min(0)).exp(), torch.zeros_like(lag))  # [H, C, C]
    scores = torch.matmul(qc, kc.transpose(-1, -2)) * intra[None, :, None]  # [B, H, n, C, C]
    numerator = torch.matmul(scores, vc)
    denominator = scores.sum(-1, keepdim=True)
    query_decay = (log_decay[:, None] * j[None, :]).exp()[None, :, :, None]  # lambda^j       [1, H, C, 1]
    key_decay = (log_decay[:, None] * (C - j)[None, :]).exp()[None, :, :, None]  # lambda^(C-j) [1, H, C, 1]
    chunk_decay = (log_decay * C).exp()  # lambda^C [H]
    state = torch.zeros(B, H, d, d, device=q.device, dtype=torch.float32)
    norm = torch.zeros(B, H, d, 1, device=q.device, dtype=torch.float32)
    inter_numerator, inter_denominator = [], []
    for c in range(n):
        qd = qc[:, :, c] * query_decay  # [B, H, C, d]
        inter_numerator.append(torch.matmul(qd, state))
        inter_denominator.append(torch.matmul(qd, norm))
        kd = kc[:, :, c] * key_decay
        state = state * chunk_decay[None, :, None, None] + torch.matmul(kd.transpose(-1, -2), vc[:, :, c])
        norm = norm * chunk_decay[None, :, None, None] + kd.sum(-2, keepdim=True).transpose(-1, -2)
    numerator = numerator + torch.stack(inter_numerator, 2)
    denominator = denominator + torch.stack(inter_denominator, 2)
    out = (numerator / (denominator + 1e-6)).reshape(B, H, T + pad, d)[:, :, :T]
    return out.transpose(1, 2)


def decayed_key_sum(k: Tensor, decays: Tensor, chunk: int = MEMORY_CHUNK) -> Tensor:
    """``z_t = sum_{i<t} lambda^(t-i) k_i`` for k [B, T, H, d] in fp32 (the normaliser of the strict read)."""
    B, T, H, d = k.shape
    C = min(chunk, T)
    pad = (-T) % C
    kh = F.pad(k.float().transpose(1, 2), (0, 0, 0, pad))  # [B, H, Tp, d]
    n = (T + pad) // C
    kc = kh.reshape(B, H, n, C, d)
    log_decay = decays.float().clamp_min(1e-30).log()
    j = torch.arange(C, device=k.device, dtype=torch.float32)
    lag = j[:, None] - j[None, :]
    intra = torch.where(lag > 0, (log_decay[:, None, None] * lag.clamp_min(0)).exp(), torch.zeros_like(lag))
    within = torch.matmul(intra[None, :, None], kc)  # [B, H, n, C, d]
    query_decay = (log_decay[:, None] * j[None, :]).exp()[None, :, :, None]
    key_decay = (log_decay[:, None] * (C - j)[None, :]).exp()[None, :, :, None]
    chunk_decay = (log_decay * C).exp()
    carry = torch.zeros(B, H, 1, d, device=k.device, dtype=torch.float32)
    across = []
    for c in range(n):
        across.append(query_decay * carry)
        carry = carry * chunk_decay[None, :, None, None] + (kc[:, :, c] * key_decay).sum(-2, keepdim=True)
    z = (within + torch.stack(across, 2)).reshape(B, H, T + pad, d)[:, :, :T]
    return z.transpose(1, 2)


@torch.library.custom_op("parameter_golf::lam_gla_forward", mutates_args=())
def _lam_gla_forward(q: Tensor, k: Tensor, v: Tensor, gate: Tensor) -> tuple[Tensor, Tensor]:
    """Fixed-length adapter for pinned fla-core 0.5.2, without its Dynamo graph break."""
    from fla.ops.simple_gla.chunk import chunk_simple_gla_fwd
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2

    cumulative = chunk_local_cumsum(gate.contiguous(), chunk_size=64, scale=RCP_LN2)
    output, _ = chunk_simple_gla_fwd(
        q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), g=cumulative,
        scale=1.0, chunk_size=64,
    )
    return output.to(q.dtype), cumulative


@_lam_gla_forward.register_fake
def _lam_gla_forward_fake(q, k, v, gate):
    return q.new_empty((*q.shape[:-1], v.shape[-1])), gate.new_empty(gate.shape)


@torch.library.custom_op("parameter_golf::lam_gla_backward", mutates_args=())
def _lam_gla_backward(
    q: Tensor, k: Tensor, v: Tensor, cumulative: Tensor, grad: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    from fla.ops.simple_gla.chunk import chunk_simple_gla_bwd
    from fla.ops.utils import chunk_local_cumsum

    dq, dk, dv, dg, _ = chunk_simple_gla_bwd(
        q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), g=cumulative,
        g_gamma=None, initial_state=None, do=grad.contiguous(), dht=None,
        scale=1.0, chunk_size=64,
    )
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True).to(cumulative)
    return dq.to(q), dk.to(k), dv.to(v), dg


@_lam_gla_backward.register_fake
def _lam_gla_backward_fake(q, k, v, cumulative, grad):
    return tuple(t.new_empty(t.shape) for t in (q, k, v, cumulative))


def _lam_gla_setup_context(ctx, inputs, output):
    q, k, v, _ = inputs
    _, cumulative = output
    ctx.save_for_backward(q, k, v, cumulative)
    ctx.mark_non_differentiable(cumulative)


def _lam_gla_autograd(ctx, grad, _):
    return _lam_gla_backward(*ctx.saved_tensors, grad)


_lam_gla_forward.register_autograd(_lam_gla_autograd, setup_context=_lam_gla_setup_context)


def _require_memory_kernel() -> None:
    if chunk_simple_gla is None:
        raise RuntimeError("CUDA LAM requires fla.ops.simple_gla.chunk_simple_gla; "
                           "install FLA or explicitly set FB_MEMORY_KERNEL=0 for the Torch reference.")


def linear_memory_read_kernel(q: Tensor, k: Tensor, v: Tensor, decays: Tensor) -> Tensor:
    """Strict fused read with bf16 operands, FLA mixed-precision states, and fp32 normaliser.

    FLA is inclusive: shift writes right and multiply by lambda for the
    strict lambda**(t-i) convention. Token gates retain learned-decay gradients.
    """
    _require_memory_kernel()
    B, T, H, _ = q.shape
    q16, k16, v16 = (x.to(torch.bfloat16) for x in (q, k, v))
    k_prev, v_prev = (F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0)) for x in (k16, v16))
    log_decay = decays.float().clamp_min(1e-30).log()
    gate = log_decay[None, None, :].expand(B, T, H).contiguous()
    if MEMORY_COMPILED:
        numerator, _ = _lam_gla_forward(q16, k_prev, v_prev, gate)
    else:
        numerator, _ = chunk_simple_gla(q16, k_prev, v_prev, g=gate, scale=1.0)
    numerator = numerator.float() * decays.float()[None, None, :, None]
    denominator = (q16.float() * decayed_key_sum(k16, decays)).sum(-1, keepdim=True)
    return numerator / (denominator + 1e-6)


def linear_memory_read_parallel(q: Tensor, k: Tensor, v: Tensor, decays: Tensor, chunk: int = MEMORY_CHUNK) -> Tensor:
    """The strict read: fused on CUDA unless ``FB_MEMORY_KERNEL=0``, chunked fp32 on CPU."""
    if MEMORY_KERNEL and q.is_cuda:
        _require_memory_kernel()
        if chunk == MEMORY_CHUNK:
            return linear_memory_read_kernel(q, k, v, decays)
    return linear_memory_read_chunked(q, k, v, decays, chunk)


class MemoryState:
    """Fast-weight state of the linear memory for sequential decoding."""

    def __init__(self, batch: int, heads: int, dim: int, device: torch.device) -> None:
        self.M = torch.zeros(batch, heads, dim, dim, device=device, dtype=torch.float32)
        self.z = torch.zeros(batch, heads, dim, device=device, dtype=torch.float32)

    def read(self, q: Tensor) -> Tensor:
        """q [B, H, d] features -> [B, H, d] values."""
        numerator = torch.einsum("bhk,bhkv->bhv", q, self.M)
        denominator = torch.einsum("bhk,bhk->bh", q, self.z)[..., None]
        return numerator / (denominator + 1e-6)

    def write(self, k: Tensor, v: Tensor, decays: Tensor) -> None:
        """k [B, H, d] features, v [B, H, d] values, decays [H]."""
        lam = decays.float()[None, :, None]
        self.M = self.M * lam[..., None] + k[..., :, None] * v.float()[..., None, :]
        self.z = self.z * lam + k


class FeedbackGPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, mode: str = "none",
                 detach: bool = False, noise: float = 0.0, mlp_hidden: int | None = None,
                 memory_head_dim: int = 128, decays: tuple[float, ...] = DEFAULT_DECAYS,
                 attn_window: int = 0, memory_layers: tuple[int, ...] | None = None) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not 0 <= noise < 1:
            raise ValueError("noise must be in [0, 1)")
        if attn_window < 0:
            raise ValueError("attn_window must be >= 0 (0 = full causal attention)")
        if memory_layers is not None:
            memory_layers = tuple(memory_layers)
            if mode != "lam":
                raise ValueError("memory_layers is only supported for mode 'lam'")
            if (not memory_layers
                    or any(type(layer) is not int or not 0 <= layer < num_layers for layer in memory_layers)
                    or tuple(sorted(set(memory_layers))) != memory_layers):
                raise ValueError("memory_layers must be sorted, unique, nonempty layer indices in [0, num_layers)")
        self.memory_layers = memory_layers
        sites = range(num_layers) if memory_layers is None else memory_layers
        read_indices = {layer: index for index, layer in enumerate(sites)} if mode == "lam" else {}
        # Static Python indices are specialized by torch.compile, with no token-time lookup allocation.
        self.memory_read_indices = tuple(read_indices.get(layer, -1) for layer in range(num_layers))
        self.mode, self.detach_state, self.noise, self.attn_window = mode, detach, noise, attn_window
        self.model_dim, self.num_layers = model_dim, num_layers
        # Trunk in the baseline's registration order.
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim, mlp_hidden) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        # Feedback parameters, registered after the trunk.
        if mode == "glu":
            self.fuse_value = Linear(model_dim, model_dim)
            self.fuse_gate = Linear(model_dim, model_dim)
        elif mode == "add":
            self.fuse_proj = Linear(model_dim, model_dim)  # zero-initialised by the trainer's "proj" rule
        elif mode in ("lam", "top"):
            if model_dim % memory_head_dim:
                raise ValueError("memory_head_dim must divide model_dim")
            self.memory_heads = model_dim // memory_head_dim
            self.memory_head_dim = memory_head_dim
            if len(decays) != self.memory_heads:
                raise ValueError(f"decays must give one value per memory head ({self.memory_heads})")
            if not all(0 < d < 1 for d in decays):
                raise ValueError("decays must lie in (0, 1)")
            self.mem_k = Linear(model_dim, model_dim)
            self.mem_v = Linear(model_dim, model_dim)
            reads = len(sites) if mode == "lam" else 1
            self.mem_q = nn.ModuleList([Linear(model_dim, model_dim) for _ in range(reads)])
            self.mem_proj = nn.ModuleList([Linear(model_dim, model_dim) for _ in range(reads)])
            self.initial_decays = tuple(decays)
            self.decay_logit = nn.Parameter(torch.empty(self.memory_heads))
            self.temperature = nn.Parameter(torch.empty(self.memory_heads))
            self.reset_extra_parameters()

    # ----------------------------------------------------------------- setup
    def reset_extra_parameters(self) -> None:
        """Decay logits and feature temperatures: not covered by the trainer's name rules."""
        if self.mode in ("lam", "top"):
            with torch.no_grad():
                self.decay_logit.copy_(torch.tensor(self.initial_decays).logit())
                self.temperature.fill_(1.0)

    def extra_scalar_parameters(self) -> list[nn.Parameter]:
        return [self.decay_logit, self.temperature] if self.mode in ("lam", "top") else []

    def feedback_matrices(self) -> list[nn.Parameter]:
        memory = ("mem_k", "mem_v", "mem_q", "mem_proj")
        names = {"glu": ("fuse_value", "fuse_gate"), "add": ("fuse_proj",), "lam": memory, "top": memory}
        out = []
        for name in names.get(self.mode, ()):
            module = getattr(self, name)
            out.extend(p for p in module.parameters() if p.ndim >= 2)
        return out

    @property
    def decays(self) -> Tensor:
        return torch.sigmoid(self.decay_logit)

    @property
    def config(self) -> dict:
        return dict(vocab_size=self.embed.num_embeddings, num_layers=self.num_layers, model_dim=self.model_dim,
                    mode=self.mode, detach=self.detach_state, noise=self.noise,
                    mlp_hidden=self.blocks[0].mlp.fc.out_features,
                    memory_head_dim=getattr(self, "memory_head_dim", 128),
                    decays=tuple(getattr(self, "initial_decays", DEFAULT_DECAYS)), attn_window=self.attn_window,
                    memory_layers=self.memory_layers)

    # -------------------------------------------------------------- parallel
    def fused_input(self, tokens: Tensor, state: Tensor | None) -> Tensor:
        """Layer-0 input [B, T, D] from tokens and the shifted top state (None: plain)."""
        e = self.embed(tokens)
        if state is None or self.mode in ("none", "lam", "top"):
            return self.norm1(e)
        previous = F.pad(state, (0, 0, 1, 0))[:, :-1].type_as(e)  # s_{t-1}, zero at t = 0
        if self.mode == "glu":
            fused = self.fuse_value(previous) * torch.sigmoid(self.fuse_gate(e))
            first = torch.zeros(tokens.size(1), device=tokens.device, dtype=torch.bool)
            first[0] = True
            fused = torch.where(first[None, :, None], e, fused)
        else:
            fused = e + self.fuse_proj(previous)
        return self.norm1(fused)

    def memory_kv(self, state: Tensor) -> tuple[Tensor, Tensor]:
        B, T, _ = state.shape
        k = self.mem_k(state).view(B, T, self.memory_heads, self.memory_head_dim)
        v = self.mem_v(state).view(B, T, self.memory_heads, self.memory_head_dim)
        return linear_memory_features(k, self.temperature), v

    def attention(self, block: Block, n: Tensor) -> Tensor:
        """The block's causal attention, optionally restricted to the last ``attn_window`` tokens."""
        if self.attn_window == 0:
            return block.attn(n)
        attn = block.attn
        B, T = n.size(0), n.size(1)
        q = attn.q(n).view(B, T, attn.num_heads, attn.head_dim)
        k = attn.k(n).view(B, T, attn.num_heads, attn.head_dim)
        v = attn.v(n).view(B, T, attn.num_heads, attn.head_dim)
        q, k = F.rms_norm(q, (attn.head_dim,)), F.rms_norm(k, (attn.head_dim,))
        q, k = attn.rotary(q), attn.rotary(k)
        mask = window_mask(T, self.attn_window, n.device)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           attn_mask=mask, scale=ATTENTION_SCALE).transpose(1, 2)
        return attn.proj(y.contiguous().view(B, T, attn.num_heads * attn.head_dim))

    def run_blocks(self, x: Tensor, memory: tuple[Tensor, Tensor, Tensor] | None) -> Tensor:
        """The stack; ``memory`` = (phi(k), v, decays) enables reads at selected LAM layers."""
        for index, block in enumerate(self.blocks):
            n = block.norm1(x)
            update = self.attention(block, n)
            read_index = self.memory_read_indices[index]
            if memory is not None and read_index >= 0:
                B, T, _ = n.shape
                q = self.mem_q[read_index](n).view(B, T, self.memory_heads, self.memory_head_dim)
                q = linear_memory_features(q, self.temperature)
                read = linear_memory_read_parallel(q, memory[0], memory[1], memory[2])
                update = update + self.mem_proj[read_index](read.reshape(B, T, -1).type_as(n))
            x = x + update
            x = x + block.mlp(block.norm2(x))
        return x

    def top_read(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Mode ``top``: (h, s) = (plain top state, top state after the memory read) from the trunk output ``x``."""
        h = self.norm2(x)
        B, T, _ = h.shape
        source = h.detach() if self.detach_state else h
        k, v = self.memory_kv(source)
        q = linear_memory_features(self.mem_q[0](h).view(B, T, self.memory_heads, self.memory_head_dim), self.temperature)
        read = linear_memory_read_parallel(q, k, v, self.decays)
        s = self.norm2(x + self.mem_proj[0](read.reshape(B, T, -1).type_as(h)))
        return h, s

    def pass_forward(self, tokens: Tensor, state: Tensor | None) -> Tensor:
        """One Jacobi pass: returns the top state ``s`` [B, T, D] (post final norm)."""
        x = self.fused_input(tokens, state)
        memory = None
        if self.mode == "lam" and state is not None:
            k, v = self.memory_kv(state)
            memory = (k, v, self.decays)
        x = self.run_blocks(x, memory)
        if self.mode == "top":
            return self.top_read(x)[1]
        return self.norm2(x)

    def loss_noread(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Mode ``top``: sum-CE of the same weights with the memory read removed (the plain trunk)."""
        if self.mode != "top":
            raise ValueError("loss_noread is defined for mode 'top'")
        x = self.run_blocks(self.fused_input(inputs, None), None)
        return self.head_loss(self.norm2(x), targets)

    def head_loss(self, s: Tensor, targets: Tensor) -> Tensor:
        logits = softcap_logits(self.proj(s))
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

    def carried(self, s: Tensor) -> Tensor:
        state = s.detach() if self.detach_state else s
        if self.training and self.noise > 0:
            state = state + (torch.rand_like(state) * 2 - 1) * self.noise
        return state

    def pass_losses(self, inputs: Tensor, targets: Tensor, passes: int) -> list[Tensor]:
        """Sum-CE of every pass; pass k >= 2 carries pass k-1's top state."""
        if self.mode in ("none", "top"):
            passes = 1
        s = self.pass_forward(inputs, None)
        losses = [self.head_loss(s, targets)]
        for _ in range(passes - 1):
            s = self.pass_forward(inputs, self.carried(s))
            losses.append(self.head_loss(s, targets))
        return losses

    def forward(self, inputs: Tensor, targets: Tensor, passes: int = 2) -> Tensor:
        losses = self.pass_losses(inputs, targets, passes)
        total = losses[0]
        if len(losses) > 1:
            total = total + torch.stack(losses[1:]).mean(0)
        return total

    # ------------------------------------------------------------ sequential
    @torch.no_grad()
    def loss_sequential(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Sum-CE under the true recurrence (token by token, KV cache, sequential feedback)."""
        B, T = inputs.shape
        device = inputs.device
        H = self.blocks[0].attn.num_heads
        d = self.blocks[0].attn.head_dim
        dtype = self.embed.weight.dtype
        k_cache = [torch.empty(B, T, H, d, device=device, dtype=dtype) for _ in self.blocks]
        v_cache = [torch.empty(B, T, H, d, device=device, dtype=dtype) for _ in self.blocks]
        freq = self.blocks[0].attn.rotary.angular_freq
        cos_all, sin_all = rotary_cos_sin(freq, torch.arange(T, device=device))
        previous = torch.zeros(B, self.model_dim, device=device, dtype=dtype)
        memory = MemoryState(B, self.memory_heads, self.memory_head_dim, device) if self.mode in ("lam", "top") else None
        layer_reads = self.mode == "lam"
        total = torch.zeros((), device=device, dtype=torch.float32)
        e_all = self.embed(inputs)
        for t in range(T):
            e = e_all[:, t]
            if self.mode == "glu" and t > 0:
                x = self.norm1(self.fuse_value(previous) * torch.sigmoid(self.fuse_gate(e)))
            elif self.mode == "add":
                x = self.norm1(e + self.fuse_proj(previous))
            else:
                x = self.norm1(e)
            cos, sin = cos_all[t][None, None], sin_all[t][None, None]
            for index, block in enumerate(self.blocks):
                n = block.norm1(x)
                attn = block.attn
                q = attn.q(n).view(B, 1, H, d)
                k = attn.k(n).view(B, 1, H, d)
                v = attn.v(n).view(B, 1, H, d)
                q, k = F.rms_norm(q, (d,)), F.rms_norm(k, (d,))
                q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
                k_cache[index][:, t] = k[:, 0]
                v_cache[index][:, t] = v[:, 0]
                start = max(0, t + 1 - self.attn_window) if self.attn_window else 0
                y = F.scaled_dot_product_attention(q.transpose(1, 2), k_cache[index][:, start:t + 1].transpose(1, 2),
                                                   v_cache[index][:, start:t + 1].transpose(1, 2), scale=ATTENTION_SCALE)
                update = attn.proj(y.transpose(1, 2).reshape(B, 1, H * d))[:, 0]
                read_index = self.memory_read_indices[index]
                if layer_reads and read_index >= 0:
                    qm = self.mem_q[read_index](n).view(B, self.memory_heads, self.memory_head_dim)
                    read = memory.read(linear_memory_features(qm, self.temperature))
                    update = update + self.mem_proj[read_index](read.reshape(B, -1).type_as(n))
                x = x + update
                x = x + block.mlp(block.norm2(x))
            h = self.norm2(x)
            s = h
            if self.mode == "top":
                qm = self.mem_q[0](h).view(B, self.memory_heads, self.memory_head_dim)
                read = memory.read(linear_memory_features(qm, self.temperature))
                s = self.norm2(x + self.mem_proj[0](read.reshape(B, -1).type_as(h)))
            total = total + self.head_loss(s, targets[:, t])
            previous = s
            if memory is not None:
                km = self.mem_k(h).view(B, self.memory_heads, self.memory_head_dim)
                vm = self.mem_v(h).view(B, self.memory_heads, self.memory_head_dim)
                memory.write(linear_memory_features(km, self.temperature), vm, self.decays)
        return total


__all__ = ["DEFAULT_DECAYS", "FeedbackGPT", "MODES", "MemoryState", "apply_rotary", "decay_mask",
           "linear_memory_features", "decayed_key_sum", "linear_memory_read_chunked", "linear_memory_read_dense", "linear_memory_read_kernel", "linear_memory_read_parallel",
           "rotary_cos_sin", "softcap_logits",
           "window_mask"]

"""pretraining/nanogpt_mini/nanogpt_mini_kda_model.py

Model classes for the KDA-mixer nanogpt-mini architecture, extracted from
``pretraining/nanogpt_mini/nanogpt_mini_gpt2vocab_kda_3to1_pm_train.py`` so the architecture is
importable without executing a training run (the script initializes
``torch.distributed`` and the training loop at module scope).

Extraction contract:
  - Class definitions match the training script byte-for-byte where possible;
    module-level env configuration (``KDA_NUM_HEADS``, ``DELTA_LAYER_INDICES``,
    ``MLP_HIDDEN``, ...) becomes constructor arguments carried by the
    checkpoint's ``model_config`` payload, so ``KDAGPT(**model_config)``
    strict-loads an exported ``final_model.pt`` with zero key mapping
    (exports exclude the training-only ``mtp_heads.*``).
  - Submodule registration order is unchanged in every class.
  - The training script wraps FLA's released kernels (``chunk_kda``,
    ``ShortConvolution``, ``FusedRMSNormGated``). Those kernels are
    Triton/CUDA-only, so this module carries its own convolution and gated
    norm (identical parameter names and shapes -> identical state-dict keys)
    plus a pure-PyTorch reference recurrence used on CPU and as the parity
    oracle for the CUDA kernels. On CUDA the full-sequence path dispatches to
    ``chunk_kda`` with exactly the training script's kernel options
    (``use_qk_l2norm/gate/beta_sigmoid_in_kernel``, ``safe_gate``,
    ``lower_bound=-5.0``, ``state_v_first=True``).

Recurrence semantics (per step t, per head h, fp32; state S is stored
value-major ``[B, H, Dv, Dk]`` to match FLA's ``state_v_first=True`` layout):

    q, k   <- l2norm(conv_silu(q_proj x)), l2norm(conv_silu(k_proj x))
              with l2norm(u) = u / sqrt(sum(u^2) + 1e-6) over the head dim
    v      <- conv_silu(v_proj x)
    g      <- lower_bound * sigmoid(exp(A_log) * (f_b(f_a x) + dt_bias))
    beta   <- sigmoid(b_proj x)
    S      <- S * exp(g)                       (decay, broadcast over Dv)
    S      <- S + beta * (v - S k) k^T         (delta-rule write)
    o      <- S (q * Dk**-0.5)
    y      <- o_proj(gated_rmsnorm(o) * sigmoid(gate))

Derived by reading FLA 0.5.2's ``fused_recurrent_kda`` kernel with the
training flags; ``tests`` assert reference == kernel on GPU.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.latent_moe import LatentMoEConfig, StableLatentMoE

KDA_SAFE_GATE_LOWER_BOUND = -5.0


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
    def __init__(self, dim: int, head_dim=128, use_rope: bool = True):
        super().__init__()
        self.use_rope = use_rope
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
        if self.use_rope:
            q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y


class ShortConv(nn.Module):
    """Depthwise causal conv + SiLU, key-compatible with FLA's ShortConvolution.

    The parameter keeps FLA's ``nn.Conv1d``-style ``[D, 1, W]`` shape and
    ``weight`` name so checkpoints written by the training script strict-load.
    The identity init used by pretraining is ``weight[:, 0, -1] = 1``: the
    newest input carries weight 1, so slot ``W-1`` of a decode cache is the
    current token.
    """

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))

    def forward(self, x: Tensor) -> Tensor:
        """Full-sequence causal conv over ``[B, T, D]`` inputs."""
        y = F.conv1d(
            x.transpose(1, 2),
            self.weight.type_as(x),
            groups=self.hidden_size,
            padding=self.kernel_size - 1,
        )[..., : x.size(1)].transpose(1, 2)
        return F.silu(y)

    def final_state(self, x: Tensor) -> Tensor:
        """The ``[B, D, W]`` cache a decode continuation needs: the last W
        raw inputs, left-padded with the zeros causal padding implies."""
        B, T, D = x.shape
        window = x[:, max(0, T - self.kernel_size):].transpose(1, 2)
        if window.size(-1) < self.kernel_size:
            window = F.pad(window, (self.kernel_size - window.size(-1), 0))
        return window.contiguous()

    def step(self, x: Tensor, cache: Tensor) -> Tensor:
        """One-token step over ``[B, D]`` input; ``cache`` updated in place."""
        shifted = torch.cat((cache[:, :, 1:], x.to(cache.dtype)[:, :, None]), dim=-1)
        cache.copy_(shifted)
        y = (shifted * self.weight[:, 0, :].type_as(shifted)).sum(-1)
        return F.silu(y).to(x.dtype)


class GatedRMSNorm(nn.Module):
    """Sigmoid-gated RMSNorm matching FLA's FusedRMSNormGated (fp32 math)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        x_f = x.float()
        rstd = torch.rsqrt(x_f.square().mean(-1, keepdim=True) + self.eps)
        y = x_f * rstd * self.weight.float() * torch.sigmoid(gate.float())
        return y.type_as(x)


def kda_decay_gate(
    decay_logits: Tensor, A_log: Tensor, dt_bias: Tensor
) -> Tensor:
    """Log-space per-dimension decay ``[..., H, Dk]`` (fp32), safe-gate form.

    ``g = lower_bound * sigmoid(exp(A_log) * (decay_logits + dt_bias))`` —
    the ``safe_gate=True, lower_bound=-5.0`` branch of FLA's KDA kernels.
    """
    H = A_log.size(0)
    biased = decay_logits.float() + dt_bias.float().view(H, -1)
    return KDA_SAFE_GATE_LOWER_BOUND * torch.sigmoid(
        A_log.float().exp()[..., None] * biased
    )


def _l2norm(x: Tensor) -> Tensor:
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)


def kda_recurrent_step(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate: Tensor,
    beta: Tensor,
    state: Tensor,
) -> Tensor:
    """One fp32 delta-rule step. ``state`` is ``[B, H, Dv, Dk]`` (value-major,
    FLA's ``state_v_first=True`` layout) and is updated IN PLACE so decode
    caches behave like the KV caches the rest of the stack mutates.

    q/k/v: ``[B, H, Dk|Dv]`` raw post-conv head projections; gate: log-space
    ``[B, H, Dk]``; beta: sigmoid write strength ``[B, H]``.
    """
    scale = q.size(-1) ** -0.5
    q = _l2norm(q.float()) * scale
    k = _l2norm(k.float())
    v = v.float()
    decayed = state * gate.exp()[:, :, None, :]
    delta = v - torch.einsum("bhvk,bhk->bhv", decayed, k)
    updated = decayed + (
        beta.float()[:, :, None, None] * delta[:, :, :, None] * k[:, :, None, :]
    )
    state.copy_(updated)
    return torch.einsum("bhvk,bhk->bhv", updated, q)


def reference_kda_recurrence(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    decay_logits: Tensor,
    beta_logits: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Pure-PyTorch oracle for ``chunk_kda`` under the training flags.

    Inputs are ``[B, T, H, D]`` post-conv projections (raw, un-normalized) and
    raw gate logits; returns ``(o, final_state)`` with ``o`` in ``v``'s dtype
    and ``final_state`` fp32 ``[B, H, Dv, Dk]``.
    """
    B, T, H, Dk = q.shape
    Dv = v.size(-1)
    gate = kda_decay_gate(decay_logits, A_log, dt_bias)
    beta = torch.sigmoid(beta_logits.float())
    state = q.new_zeros(B, H, Dv, Dk, dtype=torch.float32)
    if initial_state is not None:
        state = state + initial_state.float()
    outputs = torch.empty(B, T, H, Dv, dtype=torch.float32, device=q.device)
    for t in range(T):
        outputs[:, t] = kda_recurrent_step(
            q[:, t], k[:, t], v[:, t], gate[:, t], beta[:, t], state
        )
    return outputs.to(v.dtype), state


class KimiDeltaAttention(nn.Module):
    """Kimi-K3 KDA mixer, decode-capable and importable without Triton.

    Full-sequence forward dispatches to FLA's ``chunk_kda`` on CUDA (the
    training kernel, gradient-capable) and to the reference recurrence on
    CPU. ``step`` is pure PyTorch on every device so the rollout decode step
    stays inside one compiled graph (FLA kernels graph-break).
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 128,
        num_heads: int = 3,
        conv_size: int = 4,
        full_rank_gate: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_size = conv_size
        self.full_rank_gate = full_rank_gate
        self.projection_size = self.num_heads * head_dim

        self.q_proj = BiasFreeLinear(dim, self.projection_size)
        self.k_proj = BiasFreeLinear(dim, self.projection_size)
        self.v_proj = BiasFreeLinear(dim, self.projection_size)
        self.q_conv1d = ShortConv(self.projection_size, conv_size)
        self.k_conv1d = ShortConv(self.projection_size, conv_size)
        self.v_conv1d = ShortConv(self.projection_size, conv_size)

        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.f_a_proj = BiasFreeLinear(dim, head_dim)
        self.f_b_proj = BiasFreeLinear(head_dim, self.projection_size)
        self.dt_bias = nn.Parameter(torch.empty(self.projection_size, dtype=torch.float32))
        self.b_proj = BiasFreeLinear(dim, self.num_heads)
        if full_rank_gate:
            self.g_proj = BiasFreeLinear(dim, self.projection_size)
        else:
            self.g_a_proj = BiasFreeLinear(dim, head_dim)
            self.g_b_proj = BiasFreeLinear(head_dim, self.projection_size)
        self.o_norm = GatedRMSNorm(head_dim, eps=1e-6)
        self.o_proj = BiasFreeLinear(self.projection_size, dim)

    def _output_gate(self, x: Tensor) -> Tensor:
        gate = (
            self.g_proj(x)
            if self.full_rank_gate
            else self.g_b_proj(self.g_a_proj(x))
        )
        return gate.view(*x.shape[:-1], self.num_heads, self.head_dim)

    @torch.compiler.disable
    def _delta_rule(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        decay_logits: Tensor,
        beta_logits: Tensor,
        initial_state: Tensor | None,
        output_final_state: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """The chunked delta-rule recurrence over ``[B, T, H, D]`` inputs.

        This is the only part of the mixer Dynamo must not trace -- FLA's
        chunk kernel graph-breaks by design -- so it alone is fenced off, and
        a compiled caller still fuses the projections, short convolutions and
        gated norm around it.
        """
        B, T = q.shape[:2]
        if q.is_cuda:
            from fla.ops.kda import chunk_kda

            # Varlen call with uniform row lengths. Flattening [B, T] to
            # [1, B*T] under cu_seqlens boundaries is the same per-row math
            # (chunking and state resets are per sequence, and batching is
            # the same independence), but the TileLang kernels bake the
            # batch size into their JIT cache key while T and the sequence
            # count stay dynamic: batched calls paid a fresh ~12 s kernel
            # compile per distinct replay-shard row count (85 compiles /
            # 999 s in k3_latent_10h's first 40 min), the varlen form has
            # exactly one key per kernel, ever. Both cu_seqlens tensors are
            # built locally — device-side arange plus a CPU twin — so the
            # call adds no host-device copy.
            row_bounds_cpu = torch.arange(B + 1, dtype=torch.long) * T
            row_bounds = (
                torch.arange(B + 1, device=q.device, dtype=torch.long) * T
            )
            heads, width = self.num_heads, self.head_dim
            y, final_state = chunk_kda(
                q=q.reshape(1, B * T, heads, width),
                k=k.reshape(1, B * T, heads, width),
                v=v.reshape(1, B * T, heads, width),
                g=decay_logits.reshape(1, B * T, heads, width),
                beta=beta_logits.reshape(1, B * T, heads),
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=row_bounds,
                cu_seqlens_cpu=row_bounds_cpu,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=KDA_SAFE_GATE_LOWER_BOUND,
                state_v_first=True,
                # Pretraining ships disable_recompute=1 for backward speed on
                # 8xH100; here the grad-enabled replay/critic passes would
                # retain w/u/qg/kg/v_new/h per KDA layer at replay-shard width,
                # which does not fit the 32 GiB card at the raised shard
                # budgets. Recompute-in-backward trades that memory for time;
                # forward values are identical either way. Teacher-forced SFT
                # at 8x5120 measured no gain from disabling it either
                # (scripts/benchmark_sft_step.py, jobs 9152/9155).
                disable_recompute=False,
            )
            y = y.reshape(B, T, heads, width)
        else:
            y, final_state = reference_kda_recurrence(
                q,
                k,
                v,
                decay_logits,
                beta_logits,
                self.A_log,
                self.dt_bias,
                initial_state=initial_state,
            )
            if not output_final_state:
                final_state = None
        return y, final_state

    def forward(
        self,
        x: Tensor,
        key_valid: Tensor | None = None,
        initial_state: Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor] | None]:
        """Full-sequence mixer output, optionally with decode-ready caches.

        ``key_valid`` (bool ``[B, T]``, True = real token) zeroes the conv
        inputs at left-padded positions. That alone keeps the recurrent state
        exactly zero through the pad prefix: a causal conv over zeros emits
        k = silu(0) = 0, and the delta-rule write ``beta * (v - S k) k^T``
        vanishes with k while the decay only scales the zero state. Outputs
        at padded positions are garbage and must be discarded by the caller,
        exactly like masked dense-attention rows.

        Returns ``(y, caches)`` where ``caches`` is ``None`` unless
        ``output_final_state`` — then ``(conv_q, conv_k, conv_v, state)``
        with conv caches ``[B, D, W]`` and fp32 state ``[B, H, Dv, Dk]``.
        """
        B, T, _ = x.shape
        q_in, k_in, v_in = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if key_valid is not None:
            valid = key_valid[..., None].to(q_in.dtype)
            q_in, k_in, v_in = q_in * valid, k_in * valid, v_in * valid
        q = self.q_conv1d(q_in).view(B, T, self.num_heads, self.head_dim)
        k = self.k_conv1d(k_in).view(B, T, self.num_heads, self.head_dim)
        v = self.v_conv1d(v_in).view(B, T, self.num_heads, self.head_dim)
        decay_logits = self.f_b_proj(self.f_a_proj(x)).view(
            B, T, self.num_heads, self.head_dim
        )
        beta_logits = self.b_proj(x).float()
        y, final_state = self._delta_rule(
            q, k, v, decay_logits, beta_logits, initial_state,
            output_final_state,
        )

        y = self.o_norm(y, self._output_gate(x)).reshape(B, T, self.projection_size)
        out = self.o_proj(y)
        if not output_final_state:
            return out, None
        caches = (
            self.q_conv1d.final_state(q_in),
            self.k_conv1d.final_state(k_in),
            self.v_conv1d.final_state(v_in),
            final_state.float(),
        )
        return out, caches

    def step(
        self,
        x: Tensor,
        caches: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> Tensor:
        """One-token decode step over ``[B, 1, dim]``; caches mutate in place.

        Pure PyTorch by design: the recurrence is a handful of elementwise
        ops and two tiny einsums, so keeping it in the compiled rollout step
        graph beats calling a Triton kernel that would split the graph.
        """
        x_flat = x.squeeze(1)
        conv_q, conv_k, conv_v, state = caches
        q = self.q_conv1d.step(self.q_proj(x_flat), conv_q)
        k = self.k_conv1d.step(self.k_proj(x_flat), conv_k)
        v = self.v_conv1d.step(self.v_proj(x_flat), conv_v)
        heads = (-1, self.num_heads, self.head_dim)
        decay_logits = self.f_b_proj(self.f_a_proj(x_flat)).view(heads)
        gate = kda_decay_gate(decay_logits, self.A_log, self.dt_bias)
        beta = torch.sigmoid(self.b_proj(x_flat).float())
        o = kda_recurrent_step(
            q.view(heads), k.view(heads), v.view(heads), gate, beta, state
        ).to(x.dtype)
        y = self.o_norm(o, self._output_gate(x_flat))
        return self.o_proj(y.reshape(-1, 1, self.projection_size))




class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc = Linear(dim, hidden_dim)
        self.proj = Linear(hidden_dim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        use_kda: bool,
        mlp_hidden: int,
        delta_num_heads: int,
        delta_full_rank_gate: bool,
        delta_mlp_on_delta: bool,
        dense_position_encoding: str,
        moe_config: LatentMoEConfig | None = None,
    ):
        super().__init__()
        self.use_kda = use_kda
        if use_kda:
            self.attn = KimiDeltaAttention(
                dim,
                num_heads=delta_num_heads,
                full_rank_gate=delta_full_rank_gate,
            )
        else:
            self.attn = CausalSelfAttention(
                dim,
                use_rope=dense_position_encoding == "rope",
            )
        self.norm1 = RMSNorm(dim)
        self.use_moe = moe_config is not None
        self.use_mlp = self.use_moe or not use_kda or delta_mlp_on_delta
        if self.use_mlp:
            self.mlp = (
                StableLatentMoE(moe_config)
                if moe_config is not None
                else MLP(dim, mlp_hidden)
            )
            self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        if self.use_kda:
            mixed, _ = self.attn(self.norm1(x))
            x = x + mixed
        else:
            x = x + self.attn(self.norm1(x))
        if self.use_mlp:
            x = x + self.mlp(self.norm2(x))
        return x


class KDAGPT(nn.Module):
    """The KDA-mixer nanogpt-mini trunk, constructible from a checkpoint's
    ``model_config`` payload: ``KDAGPT(**model_config)``.

    ``mla_*`` and ``delta_residual_rank`` ride along in every checkpoint's
    config regardless of the attention types actually used; they are accepted
    and validated but only the ``kda``/``mha`` combination is implemented
    (the k3 quality campaign's configuration).
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        mlp_hidden: int,
        delta_num_heads: int,
        delta_layer_indices: list[int],
        delta_attention_type: str = "kda",
        delta_full_rank_gate: bool = False,
        delta_mlp_on_delta: bool = False,
        dense_attention_type: str = "mha",
        dense_position_encoding: str = "rope",
        mla_q_rank: int | None = None,
        mla_kv_rank: int | None = None,
        mla_qk_nope_dim: int | None = None,
        mla_shared_qk_dim: int | None = None,
        mla_v_head_dim: int | None = None,
        delta_residual_rank: int | None = None,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        moe_latent_dim: int = 128,
        moe_expert_hidden: int = 256,
        moe_shared_hidden: int = 64,
        moe_num_shared_experts: int = 2,
        moe_layer_indices: list[int] | None = None,
        tokenizer_provenance: dict | None = None,
        pretraining_data_path: str | None = None,
    ):
        super().__init__()
        # Tokenizer identity is checkpoint metadata rather than an architectural
        # hyperparameter. Accept it so the self-describing model_config can be
        # passed through every post-training constructor unchanged.
        del tokenizer_provenance, pretraining_data_path
        if delta_attention_type != "kda":
            raise NotImplementedError(
                f"delta_attention_type={delta_attention_type!r}: only 'kda' "
                "has a decode-capable implementation"
            )
        if dense_attention_type != "mha":
            raise NotImplementedError(
                f"dense_attention_type={dense_attention_type!r}: only 'mha' "
                "has a decode-capable implementation"
            )
        if dense_position_encoding not in {"rope", "none"}:
            raise ValueError(
                "dense_position_encoding must be 'rope' or 'none', got "
                f"{dense_position_encoding!r}"
            )
        if delta_residual_rank is not None:
            raise NotImplementedError(
                "delta_residual_rank is a gdn2_kda_erase parameter; "
                "only plain KDA is implemented"
            )
        delta_layers = set(delta_layer_indices)
        if not delta_layers <= set(range(num_layers)):
            raise ValueError(
                f"delta_layer_indices {sorted(delta_layers)} outside "
                f"0..{num_layers - 1}"
            )
        if moe_num_experts < 0:
            raise ValueError("moe_num_experts must be nonnegative")
        moe_layers = (
            set(range(num_layers))
            if moe_num_experts and moe_layer_indices is None
            else set(moe_layer_indices or ())
        )
        if not moe_layers <= set(range(num_layers)):
            raise ValueError(
                f"moe_layer_indices {sorted(moe_layers)} outside "
                f"0..{num_layers - 1}"
            )
        if not moe_num_experts and moe_layers:
            raise ValueError(
                "moe_layer_indices requires moe_num_experts to be positive"
            )
        moe_config = None
        if moe_num_experts:
            moe_config = LatentMoEConfig(
                model_dim=model_dim,
                latent_dim=moe_latent_dim,
                routed_hidden_dim=moe_expert_hidden,
                num_routed_experts=moe_num_experts,
                experts_per_token=moe_top_k,
                shared_hidden_dim=moe_shared_hidden,
                num_shared_experts=moe_num_shared_experts,
            )
        self.delta_layer_indices = sorted(delta_layers)
        self.moe_layer_indices = sorted(moe_layers)
        self.dense_position_encoding = dense_position_encoding
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.Sequential(*[
            Block(
                model_dim,
                use_kda=layer_idx in delta_layers,
                mlp_hidden=mlp_hidden,
                delta_num_heads=delta_num_heads,
                delta_full_rank_gate=delta_full_rank_gate,
                delta_mlp_on_delta=delta_mlp_on_delta,
                dense_position_encoding=dense_position_encoding,
                moe_config=moe_config if layer_idx in moe_layers else None,
            )
            for layer_idx in range(num_layers)
        ])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def hidden_states(self, inputs: Tensor) -> Tensor:
        return self.norm2(self.blocks(self.norm1(self.embed(inputs))))

    def project_logits(self, hidden: Tensor) -> Tensor:
        logits = self.proj(hidden).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()

    def logits(self, inputs: Tensor, positions: Tensor | None = None) -> Tensor:
        """Softcapped next-token logits, at selected positions when given.

        Greedy decoding reads one position per row while the trunk must still
        see the whole prefix. The vocabulary projection is twice the trunk's
        parameter count, so gathering before projecting keeps it off every
        position the decoder is not about to sample from.
        """

        hidden = self.hidden_states(inputs)
        if positions is not None:
            if positions.shape != inputs.shape[:1]:
                raise ValueError("positions must hold one index per row")
            hidden = torch.gather(
                hidden,
                1,
                positions[:, None, None].expand(-1, 1, hidden.shape[-1]),
            )
        return self.project_logits(hidden)

    def forward(self, inputs: Tensor, targets: Tensor):
        logits = self.project_logits(self.hidden_states(inputs))
        return F.cross_entropy(
            logits.view(targets.numel(), -1), targets.view(-1), reduction="sum"
        )

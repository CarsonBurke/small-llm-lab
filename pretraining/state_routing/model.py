"""Exact token-major top-1 routing over shared, three-head KDA memory banks.

Only matrix memories are shared. Projections, short-convolution histories,
normalization, and dense attention retain their reference layer ownership.
All state transitions are functional and differentiable; activation
checkpointing recomputes transitions without truncating their gradients.
"""

from __future__ import annotations

import time
from types import MethodType

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from fla.ops.kda import chunk_kda
except ImportError:  # pragma: no cover - production training supplies fla-core
    chunk_kda = None


class RunDeadlineReached(Exception):
    """Stop between operations without applying a partial optimizer update."""


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def dense_step(attn, x, key_cache, value_cache, position):
    """Reference RoPE MHA, expressed as a differentiable cached token step."""
    batch = x.size(0)
    shape = (batch, attn.num_heads, attn.head_dim)
    q, k, v = attn.q(x).view(shape), attn.k(x).view(shape), attn.v(x).view(shape)
    q, k = norm(q), norm(k)
    theta = position.float() * attn.rotary.angular_freq
    cosine, sine = theta.cos(), theta.sin()

    def rotate(z):
        a, b = z.float().chunk(2, -1)
        return torch.cat((a * cosine + b * sine, -a * sine + b * cosine), -1).to(
            z.dtype
        )

    q, k = rotate(q), rotate(k)
    slot = position.view(1, 1, 1, 1).expand(batch, attn.num_heads, 1, attn.head_dim)
    key_cache = key_cache.scatter(2, slot, k.unsqueeze(2))
    value_cache = value_cache.scatter(2, slot, v.unsqueeze(2))
    mask = torch.arange(key_cache.size(2), device=x.device) <= position
    y = F.scaled_dot_product_attention(
        q.unsqueeze(2),
        key_cache,
        value_cache,
        attn_mask=mask.view(1, 1, 1, -1),
        scale=0.12,
    ).reshape(batch, attn.num_heads * attn.head_dim)
    return attn.proj(y), key_cache, value_cache


def routed_kda_step(attn, x, pool, history, fixed_index=None):
    batch, banks, heads, width, _ = pool.shape
    probabilities = F.linear(x.float(), attn.state_router.weight.float()).softmax(-1)
    index = probabilities.argmax(-1)
    if fixed_index is not None:
        index = torch.full_like(index, fixed_index)
    confidence = probabilities.gather(-1, index[:, None])
    if fixed_index is not None:
        # Oracle-only mode: one fixed private bank and ordinary KDA amplitude.
        confidence = torch.ones_like(confidence)
    selected = pool[torch.arange(batch, device=x.device), index]
    projected = torch.stack((attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)))
    history = torch.cat((history, projected.unsqueeze(-1)), dim=-1)
    weights = torch.stack(
        (attn.q_conv1d.weight, attn.k_conv1d.weight, attn.v_conv1d.weight)
    )
    convolved = F.silu(
        (history.float() * weights[:, None, :, 0, :].float()).sum(-1)
    ).to(x.dtype)
    q, k, v = convolved.reshape(3, batch, heads, width).unbind(0)
    q = q.float()
    # FLA uses sqrt(sum(square) + 1e-6), not normalize's clamp(norm, eps).
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
    logits = attn.f_b_proj(attn.f_a_proj(x)).view(batch, heads, width).float()
    gate = -5.0 * torch.sigmoid(
        attn.A_log.float().exp()[None, :, None]
        * (logits + attn.dt_bias.float().view(heads, width))
    )
    beta = attn.b_proj(x).float().sigmoid()
    decayed = selected * gate.exp().unsqueeze(-2)
    delta = v.float() - (decayed * k.unsqueeze(-2)).sum(-1)
    updated = decayed + beta[..., None, None] * delta.unsqueeze(-1) * k.unsqueeze(-2)
    output = (updated * (q * width**-0.5).unsqueeze(-2)).sum(-1).to(x.dtype)
    output_gate = (
        attn.g_proj(x) if hasattr(attn, "g_proj") else attn.g_b_proj(attn.g_a_proj(x))
    ).view(batch, heads, width)
    # Same sigmoid RMS gate as FLA, without its eager-only module boundary.
    normalized = output.float() * torch.rsqrt(
        output.float().square().mean(-1, keepdim=True) + 1e-6
    )
    normalized = normalized * attn.o_norm.weight.float() * output_gate.float().sigmoid()
    mixed = attn.o_proj(normalized.to(x.dtype).reshape(batch, heads * width))
    mixed = mixed * confidence.to(x.dtype)
    pool = pool.scatter(
        1,
        index[:, None, None, None, None].expand(-1, 1, heads, width, width),
        updated[:, None],
    )
    return mixed, pool, history[..., 1:], probabilities, F.one_hot(index, banks).float()


def routed_kda_sequence(attn, x, pool, fixed_index=None):
    """Run routed KDA with the fused variable-length chunk kernel.

    This is the efficient execution path.  Tokens assigned to the same bank
    are gathered in causal order and processed as independent variable-length
    sequences by FLA's chunk kernel.  Unselected tokens are omitted from the
    recurrent scan (an identity transition), while their short-convolution
    context is retained because convolution is performed before gathering.

    The bank state is threaded between KDA layers at sequence boundaries.  It
    is the scan-compatible interpretation of the shared bank pool; the exact
    token-major executor remains available for numerical contracts.
    """
    if chunk_kda is None:
        raise RuntimeError("routed_kda_sequence requires fla-core")
    batch, length, _ = x.shape
    banks = pool.size(1)
    heads = pool.size(2)
    width = pool.size(3)
    probabilities = F.linear(x.float(), attn.state_router.weight.float()).softmax(-1)
    routes = probabilities.argmax(-1)
    if fixed_index is not None:
        routes = torch.full_like(routes, fixed_index)

    projected = []
    for projection, conv in (
        (attn.q_proj, attn.q_conv1d),
        (attn.k_proj, attn.k_conv1d),
        (attn.v_proj, attn.v_conv1d),
    ):
        conv_out, _ = conv(projection(x), output_final_state=False)
        projected.append(conv_out.view(batch, length, heads, width))
    q, k, v = projected
    g = attn.f_b_proj(attn.f_a_proj(x)).view(batch, length, heads, width)
    beta = attn.b_proj(x).float()
    gate = (
        attn.g_proj(x) if hasattr(attn, "g_proj") else attn.g_b_proj(attn.g_a_proj(x))
    ).view(batch, length, heads, width)

    # Build B*banks causal segments.  Sorting by (bank, batch, token) makes
    # one fused call sufficient; cu_seqlens keeps each bank/batch trajectory
    # independent and lets the kernel skip unselected tokens entirely.
    segment_indices = []
    segment_lengths = []
    segment_states = []
    segment_ids = []
    for bank in range(banks):
        for b in range(batch):
            local = torch.nonzero(routes[b] == bank, as_tuple=False).flatten()
            segment_lengths.append(local.numel())
            if local.numel():
                segment_indices.append(local + b * length)
            segment_states.append(pool[b, bank])
            segment_ids.append(b * banks + bank)
    device = x.device
    if not segment_indices:
        return (
            torch.zeros_like(x),
            pool,
            probabilities,
            F.one_hot(routes, banks).float(),
        )
    flat_indices = torch.cat(segment_indices)
    flat_q = q.reshape(batch * length, heads, width).index_select(0, flat_indices)
    flat_k = k.reshape(batch * length, heads, width).index_select(0, flat_indices)
    flat_v = v.reshape(batch * length, heads, width).index_select(0, flat_indices)
    flat_g = g.reshape(batch * length, heads, width).index_select(0, flat_indices)
    flat_beta = beta.reshape(batch * length, heads).index_select(0, flat_indices)
    flat_gate = gate.reshape(batch * length, heads, width).index_select(0, flat_indices)
    cu = torch.zeros(len(segment_lengths) + 1, device=device, dtype=torch.long)
    cu[1:] = torch.tensor(segment_lengths, device=device, dtype=torch.long).cumsum(0)
    initial = torch.stack(segment_states)
    raw, final = chunk_kda(
        q=flat_q.unsqueeze(0),
        k=flat_k.unsqueeze(0),
        v=flat_v.unsqueeze(0),
        g=flat_g.unsqueeze(0),
        beta=flat_beta.unsqueeze(0),
        A_log=attn.A_log,
        dt_bias=attn.dt_bias,
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        state_v_first=True,
        disable_recompute=True,
        cu_seqlens=cu,
    )
    raw = raw.squeeze(0)
    mixed_selected = attn.o_proj(attn.o_norm(raw, flat_gate).reshape(-1, heads * width))
    confidence = (
        probabilities.reshape(batch * length, banks)
        .gather(1, routes.reshape(-1, 1))
        .squeeze(1)
        .index_select(0, flat_indices)
    )
    if fixed_index is not None:
        confidence = torch.ones_like(confidence)
    mixed_selected = mixed_selected * confidence.to(mixed_selected.dtype).unsqueeze(-1)
    mixed_flat = torch.zeros(
        batch * length, x.size(-1), device=device, dtype=mixed_selected.dtype
    )
    mixed_flat = mixed_flat.index_copy(0, flat_indices, mixed_selected)
    old = torch.stack(segment_states)
    delta = final - old
    updates = torch.zeros(
        batch * banks, heads, width, width, device=device, dtype=pool.dtype
    )
    updates = updates.index_add(
        0, torch.tensor(segment_ids, device=device), delta.to(pool.dtype)
    )
    new_pool = pool + updates.view(batch, banks, heads, width, width)
    return (
        mixed_flat.view(batch, length, -1),
        new_pool,
        probabilities,
        F.one_hot(routes, banks).float(),
    )


class SharedStateExecutor:
    """Execution policy, deliberately not another owner of model parameters."""

    def __init__(
        self,
        model,
        *,
        banks=12,
        checkpoint_tokens=16,
        balance_weight=0.01,
        compile_step=True,
        checkpointing=True,
        deadline=None,
        fixed_indices=None,
        fast_sequence=False,
    ):
        self.model = model
        self.banks = banks
        self.checkpoint_tokens = checkpoint_tokens
        self.balance_weight = balance_weight
        self.deadline = deadline
        self.fixed_indices = fixed_indices
        self.fast_sequence = fast_sequence
        self.checkpointing = checkpointing
        self.kda_blocks = [block for block in model.blocks if block.use_kda]
        self.dense_blocks = [block for block in model.blocks if not block.use_kda]
        if banks <= 0 or checkpoint_tokens <= 0:
            raise ValueError("bank count and checkpoint interval must be positive")
        if fixed_indices is not None and (
            len(fixed_indices) != len(self.kda_blocks)
            or any(not 0 <= index < banks for index in fixed_indices)
        ):
            raise ValueError("fixed routes must specify one valid bank per KDA site")
        self.token_step = (
            torch.compile(self.step, fullgraph=True, dynamic=False)
            if compile_step
            else self.step
        )
        # The final block is dense and is downstream of the last shared-state
        # writer.  It therefore has no effect on any recurrent state and can
        # remain sequence-batched, like the reference implementation.
        self.prefix_token_step = (
            torch.compile(self.prefix_step, fullgraph=True, dynamic=False)
            if compile_step
            else self.prefix_step
        )
        self.terminal = (
            torch.compile(self.loss, fullgraph=True, dynamic=False)
            if compile_step
            else self.loss
        )
        self.last_route_counts = None
        self.microbatches_started = 0

    def check_deadline(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise RunDeadlineReached("training job wall-clock budget reached")

    def initial_state(self, batch, length, device):
        attn = self.kda_blocks[0].attn
        pool = torch.zeros(
            batch,
            self.banks,
            attn.num_heads,
            attn.head_dim,
            attn.head_dim,
            device=device,
        )
        histories = tuple(
            torch.zeros(
                3,
                batch,
                b.attn.projection_size,
                b.attn.q_conv1d.weight.size(-1) - 1,
                device=device,
                dtype=torch.bfloat16,
            )
            for b in self.kda_blocks
        )
        keys = tuple(
            torch.zeros(
                batch,
                b.attn.num_heads,
                length,
                b.attn.head_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            for b in self.dense_blocks
        )
        return pool, histories, keys, tuple(torch.zeros_like(k) for k in keys)

    def step(self, x, pool, histories, keys, values, position):
        new_histories, new_keys, new_values, probabilities, counts = [], [], [], [], []
        kd, dense = 0, 0
        for block in self.model.blocks:
            normalized = block.norm1(x)
            if block.use_kda:
                fixed = None if self.fixed_indices is None else self.fixed_indices[kd]
                mixed, pool, history, prob, count = routed_kda_step(
                    block.attn, normalized, pool, histories[kd], fixed
                )
                new_histories.append(history)
                probabilities.append(prob)
                counts.append(count)
                kd += 1
            else:
                mixed, key, value = dense_step(
                    block.attn, normalized, keys[dense], values[dense], position
                )
                new_keys.append(key)
                new_values.append(value)
                dense += 1
            x = x + mixed
            if block.use_mlp:
                x = x + block.mlp(block.norm2(x))
        return (
            x,
            pool,
            tuple(new_histories),
            tuple(new_keys),
            tuple(new_values),
            torch.stack(probabilities),
            torch.stack(counts),
        )

    def prefix_step(self, x, pool, histories, keys, values, position):
        """Advance only through the last shared-state writer.

        The final block is deliberately omitted: it is a dense block after
        all KDA writes, so evaluating it here would needlessly serialize a
        sequence-independent operation.
        """
        new_histories, new_keys, new_values, probabilities, counts = [], [], [], [], []
        kd, dense = 0, 0
        for block in self.model.blocks[:-1]:
            normalized = block.norm1(x)
            if block.use_kda:
                fixed = None if self.fixed_indices is None else self.fixed_indices[kd]
                mixed, pool, history, prob, count = routed_kda_step(
                    block.attn, normalized, pool, histories[kd], fixed
                )
                new_histories.append(history)
                probabilities.append(prob)
                counts.append(count)
                kd += 1
            else:
                mixed, key, value = dense_step(
                    block.attn, normalized, keys[dense], values[dense], position
                )
                new_keys.append(key)
                new_values.append(value)
                dense += 1
            x = x + mixed
            if block.use_mlp:
                x = x + block.mlp(block.norm2(x))
        # Preserve the state tuple shape; the final dense cache is unused by
        # the prefix and is intentionally carried through unchanged.
        new_keys.extend(keys[dense:])
        new_values.extend(values[dense:])
        return (
            x,
            pool,
            tuple(new_histories),
            tuple(new_keys),
            tuple(new_values),
            torch.stack(probabilities),
            torch.stack(counts),
        )

    def tail(self, hidden):
        """Run the final dense block over the complete sequence."""
        block = self.model.blocks[-1]
        hidden = hidden + block.attn(block.norm1(hidden))
        if block.use_mlp:
            hidden = hidden + block.mlp(block.norm2(hidden))
        return hidden

    def loss(self, hidden, targets):
        logits = self.model.proj(self.model.norm2(hidden)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")

    def chunk(self, inputs, targets, positions, pool, histories, keys, values):
        embedded = self.model.norm1(self.model.embed(inputs))
        outputs, probabilities, counts = [], [], []
        for t in range(inputs.size(1)):
            hidden, pool, histories, keys, values, prob, count = self.prefix_token_step(
                embedded[:, t], pool, histories, keys, values, positions[t]
            )
            outputs.append(hidden)
            probabilities.append(prob)
            counts.append(count)
        # Block 7 has no state side effects; restore full-sequence execution.
        loss = self.terminal(self.tail(torch.stack(outputs, 1)), targets)
        loads = torch.stack(counts).sum((0, 1, 2))
        if self.model.training and self.balance_weight and self.fixed_indices is None:
            mean_prob = torch.stack(probabilities).mean((0, 1, 2))
            frequency = loads / (inputs.numel() * len(self.kda_blocks))
            loss = (
                loss
                + self.balance_weight
                * inputs.numel()
                * self.banks
                * (mean_prob * frequency).sum()
            )
        return loss, pool, histories, keys, values, loads

    def forward(self, inputs, targets):
        self.check_deadline()
        self.microbatches_started += 1
        if self.fast_sequence:
            return self.sequence_forward(inputs, targets)
        state = self.initial_state(inputs.size(0), inputs.size(1), inputs.device)
        positions = torch.arange(inputs.size(1), device=inputs.device)
        total = torch.zeros((), device=inputs.device)
        counts = torch.zeros(self.banks, device=inputs.device)
        for start in range(0, inputs.size(1), self.checkpoint_tokens):
            stop = start + self.checkpoint_tokens
            args = (
                inputs[:, start:stop],
                targets[:, start:stop],
                positions[start:stop],
                *state,
            )
            result = (
                checkpoint(
                    self.chunk, *args, use_reentrant=False, preserve_rng_state=False
                )
                if self.checkpointing
                and self.model.training
                and torch.is_grad_enabled()
                else self.chunk(*args)
            )
            loss, *state, loads = result
            total = total + loss
            counts = counts + loads.detach()
        self.last_route_counts = counts
        return total

    def sequence_forward(self, inputs, targets):
        """Sequence-parallel scan path used by training.

        Unlike the original token executor this keeps the dense attention and
        MLPs fully batched and dispatches each KDA site through the fused FLA
        chunk kernel.  The shared bank pool is carried between KDA sites as a
        terminal scan state, which is the only layout compatible with the
        sequence-parallel recurrent kernel.
        """
        if chunk_kda is None:
            raise RuntimeError("fast_sequence requires fla-core")
        x = self.model.norm1(self.model.embed(inputs))
        batch, length = inputs.shape
        attn = self.kda_blocks[0].attn
        pool = torch.zeros(
            batch,
            self.banks,
            attn.num_heads,
            attn.head_dim,
            attn.head_dim,
            device=inputs.device,
            dtype=torch.float32,
        )
        probabilities, counts = [], []
        kd = 0
        for block in self.model.blocks:
            normalized = block.norm1(x)
            if block.use_kda:
                fixed = None if self.fixed_indices is None else self.fixed_indices[kd]
                mixed, pool, prob, count = routed_kda_sequence(
                    block.attn, normalized, pool, fixed
                )
                probabilities.append(prob)
                counts.append(count)
                kd += 1
            else:
                mixed = block.attn(normalized)
            x = x + mixed
            if block.use_mlp:
                x = x + block.mlp(block.norm2(x))
        logits = self.model.proj(self.model.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        total = F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="sum"
        )
        stacked_counts = torch.stack(counts)
        loads = stacked_counts.sum((0, 1, 2))
        if self.model.training and self.balance_weight and self.fixed_indices is None:
            mean_prob = torch.stack(probabilities).mean((0, 1, 2))
            frequency = loads / (inputs.numel() * len(self.kda_blocks))
            total = (
                total
                + self.balance_weight
                * inputs.numel()
                * self.banks
                * (mean_prob * frequency).sum()
            )
        self.last_route_counts = loads.detach()
        return total


def install_shared_state_routing(model, *, seed=1337, **options):
    """Call after reference initialization; routers never perturb trunk RNG."""
    banks = options.get("banks", 12)
    for layer, block in enumerate(model.blocks):
        if block.use_kda:
            attn = block.attn
            router = nn.Linear(
                attn.q_proj.in_features,
                banks,
                bias=False,
                device=attn.q_proj.weight.device,
            )
            generator = torch.Generator(device=router.weight.device).manual_seed(
                seed + 70_000 + layer
            )
            with torch.no_grad():
                router.weight.normal_(std=0.02, generator=generator)
            attn.state_router = router
    executor = SharedStateExecutor(model, **options)
    model.shared_state_executor = executor

    def forward(self, inputs, targets):
        return self.shared_state_executor.forward(inputs, targets)

    model.forward = MethodType(forward, model)
    return executor

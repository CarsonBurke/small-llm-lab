"""pretraining/nanogpt_mini/nanogpt_mini_tree_model.py

Tree-pointer attention: ``O(K log N)`` per query, dense at ``K = N``.

Per layer one shared K/V head (pointer-GQA: every query head attends the
same gathered rows) and a binary tree over absolute positions ``[0, T)``.
A ROW is a token (``k_i, v_i``, count 1) or an interval summary (mean
``k``, mean ``v``, count): the ``T - 1`` full tree nodes plus, for every
query ``t`` and depth ``d``, the partial node containing ``t`` truncated to
``[start_d(t), t]`` ("edge" rows, a function of ``(t, d)`` only). All rows
live in one table built once per layer in ``O(T log T)``; keys carry
rotary before pooling.

Per query ``t`` and pointer ``j`` of ``K`` (shared by the query heads): a
root-to-leaf descent. At the node ``[lo, hi)`` with ``mid``: if ``mid > t``
the right child is empty and the descent goes left with no decision;
otherwise the bit ``b_d ~ Bernoulli(sigmoid(z_d))`` picks right (1) or
left (0), with

    z_d = head(x_t, f_d) + gain * scale * q_r . (kbar_right - kbar_left)

where ``head`` is the shared-depth controller of ``nanogpt_mini_iptr_model``
(features ``f_d``: ``log2(t+1)``, ``d``, ``log2 size``, ``log2(t - mid + 1)``)
and ``q_r`` a routing query (own projection, rotary at ``t``) against the
two children's pooled keys: the search reads the array. The sibling NOT
taken enters the candidate set as its summary row with log prior
``log pi(path so far) + log pi(flip_d)``; the leaf token enters with
``log pi(path)``. Attention = one softmax over the ``K (D + 1)`` rows with
score ``q_h . k_row * scale + log count_row + log prior_row``: every query
sees its whole context at coarse resolution and ``K`` paths at token
resolution. Only the prior term reaches the controller / routing query /
pooled keys (finite-difference credit: mass landing on a sibling summary
says "descend here"). Bits are sampled in training and eval alike.

Backends: ``fused`` (Triton kernels of ``nanogpt_mini_pgqa_kernel`` over
the extended row table) and ``gather`` (explicit, CPU-runnable reference).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pretraining.nanogpt_mini.nanogpt_mini_model import MLP, Linear, RMSNorm, Rotary
from pretraining.nanogpt_mini.nanogpt_mini_iptr_model import IntervalPointerHead
from pretraining.nanogpt_mini.nanogpt_mini_pgqa_model import GatherRows, MASKED_SCORE, SCORE_SCALE

BACKENDS = ("fused", "gather")
# routing features per decision (``IntervalPointerHead`` sees NUM_FEATURES = 4):
# log2(t + 1), depth d, level (node size 2^level), log2(t - mid + 1)
LEVEL_FEATURE = 2


def tree_depth(seq_len: int) -> int:
    if seq_len <= 0 or seq_len & (seq_len - 1):
        raise ValueError(f"seq_len must be a power of two, got {seq_len}")
    return seq_len.bit_length() - 1


def node_row(lo: Tensor, level: int, seq_len: int) -> Tensor:
    """Row id of the full node ``[lo, lo + 2^level)``. Level-``l`` nodes
    (``l >= 1``) occupy ``T/2^l`` rows after the tokens, ordered by level:
    offset ``T + sum_{m < l} T / 2^m - T = T + (T - T / 2^(l-1))``. Level 0
    is the token itself."""
    if level == 0:
        return lo
    offset = seq_len + (seq_len - (seq_len >> (level - 1)))
    return offset + (lo >> level)


def edge_row(t: Tensor, depth: int, seq_len: int, depths: int) -> Tensor:
    """Row id of the partial node containing ``t`` at ``depth`` (size
    ``2^(D - depth)``), truncated to ``[start, t]``: rows ``T·(D+1)`` after
    the ``2T - 1`` tokens + full nodes."""
    return (2 * seq_len - 1) + t * (depths + 1) + depth


def num_rows(seq_len: int) -> int:
    depths = tree_depth(seq_len)
    return 2 * seq_len - 1 + seq_len * (depths + 1)


def build_row_table(k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """``k``/``v``: ``[B, T, Dh]`` (k rotary'd). Returns ``(table, log_count)``:
    ``table`` ``[B, R, 2 Dh]`` (mean k | mean v per row, in ``k``'s dtype),
    ``log_count`` ``[R]`` fp32. Tokens, then full nodes by level, then edge
    rows ``(t, d)``."""
    B, T, Dh = k.shape
    D = tree_depth(T)
    kv = torch.cat((k, v), -1)
    acc = torch.promote_types(kv.dtype, torch.float32)
    prefix = torch.cumsum(kv.to(acc), dim=1)                                  # P[j] = sum_{i <= j}
    rows = [kv]
    counts = [torch.ones(T, device=k.device)]
    for level in range(1, D + 1):
        size = 1 << level
        sums = kv.to(acc).view(B, T // size, size, 2 * Dh).sum(2)
        rows.append((sums / size).to(kv.dtype))
        counts.append(torch.full((T // size,), float(size), device=k.device))
    # edge rows: for t and depth d, start = t with the low (D - d) bits cleared
    t = torch.arange(T, device=k.device)
    edge = []
    edge_counts = []
    for d in range(D + 1):
        start = (t >> (D - d)) << (D - d)
        count = (t - start + 1).to(acc)
        lower = prefix[:, (start - 1).clamp_min(0)] * (start > 0).view(1, T, 1)
        mean = (prefix[:, t] - lower) / count.view(1, T, 1)
        edge.append(mean.to(kv.dtype))
        edge_counts.append(count)
    edge = torch.stack(edge, 2).reshape(B, T * (D + 1), 2 * Dh)               # (t, d) order
    edge_counts = torch.stack(edge_counts, 1).reshape(-1)
    table = torch.cat(rows + [edge], 1)
    log_count = torch.log(torch.cat(counts + [edge_counts])).to(torch.float32)
    return table, log_count


@torch.compiler.disable
def fused_rows_attend(q: Tensor, table: Tensor, rows: Tensor, score_bias: Tensor) -> Tensor:
    """``attend_gather`` on the Triton kernels (a compile boundary, as in
    the pgqa model): ``table`` ``[B, R, 2 Dh]``, ``rows`` long ``[B, T, P]``
    (``-1`` masked), ``score_bias`` fp32 ``[B, T, P]``."""
    from pretraining.nanogpt_mini.nanogpt_mini_pgqa_kernel import fused_attend
    return fused_attend(q, table, rows.to(torch.int32), score_bias.to(torch.float32), SCORE_SCALE)


class TreePointerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        num_pointers: int,
        width: int,
        head_dim: int = 128,
        backend: str = "fused",
        local_window: int = 0,
    ):
        super().__init__()
        if dim % head_dim or dim < head_dim:
            raise ValueError(f"dim {dim} must be a positive multiple of head_dim {head_dim}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.depths = tree_depth(seq_len)
        self.num_pointers = num_pointers
        self.backend = backend
        if local_window < 0 or local_window > seq_len:
            raise ValueError(f"local_window must be in [0, seq_len], got {local_window}")
        # always-on real tokens [t - W + 1, t] with prior mass 1 (log 1 = 0),
        # as iptr's PTR_LOCAL: the tree spends its budget on the far context
        self.local_window = local_window
        hdim = self.num_heads * head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, head_dim)
        self.v = Linear(dim, head_dim)
        self.qr = Linear(dim, head_dim)          # routing query, shared by the heads
        self.proj = Linear(hdim, dim)
        # one shared-depth controller "head" (num_heads = 1: pointers are shared by the query heads)
        self.head = IntervalPointerHead(dim, 1, num_pointers, width)
        self.route_gains = nn.Parameter(torch.ones(num_pointers))
        self.rotary = Rotary(head_dim)

    @property
    def candidates_per_query(self) -> int:
        return self.num_pointers * (self.depths + 1) + self.local_window

    def qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        B, T, H, D = x.size(0), x.size(1), self.num_heads, self.head_dim
        q = self.q(x).view(B, T, H, D)
        k = self.k(x).view(B, T, 1, D)
        v = self.v(x).view(B, T, D)
        qr = self.qr(x).view(B, T, 1, D)
        q, k, qr = F.rms_norm(q, (D,)), F.rms_norm(k, (D,)), F.rms_norm(qr, (D,))
        return self.rotary(q), self.rotary(k).squeeze(2), v, self.rotary(qr).squeeze(2)

    def route(self, x: Tensor, qr: Tensor, table: Tensor) -> tuple[Tensor, Tensor]:
        """Root-to-leaf descents of the ``K`` pointers. Returns ``(rows,
        log_prior)`` ``[B, T, K (D + 1)]``: row ids (``-1`` = no candidate)
        and each candidate's log prior; slot ``j (D + 1) + d`` holds pointer
        ``j``'s sibling summary at depth ``d``, slot ``j (D + 1) + D`` its leaf."""
        B, T, K, D, Dh = x.size(0), x.size(1), self.num_pointers, self.depths, self.head_dim
        R = table.size(1)
        u = self.head.hidden_input(x)                                          # [B, T, 1, K, R], >= fp32
        ft = u.dtype
        t = torch.arange(T, device=x.device).view(1, T, 1)
        lo = torch.zeros(B, T, K, dtype=torch.long, device=x.device)
        path_lp = torch.zeros(B, T, K, dtype=ft, device=x.device)
        log_len = torch.log2(t.to(ft) + 1).expand(B, T, K)
        flat_kv = table.reshape(B * R, 2 * Dh)
        batch_off = (torch.arange(B, device=x.device) * R).view(B, 1, 1)
        qr_k = qr.to(ft).unsqueeze(2)                                          # [B, T, 1, Dh]
        gain = (self.route_gains.to(ft) * SCORE_SCALE).view(1, 1, K)
        rows, lps = [], []
        for d in range(D):
            level = D - d                       # current node size 2^level
            half = 1 << (level - 1)
            mid = lo + half
            decide = mid <= t                   # right child nonempty
            end = lo + (1 << level)
            left = node_row(lo, level - 1, T)                                   # full: mid <= t + 1 whenever decide
            right = torch.where(end <= t + 1, node_row(mid, level - 1, T), edge_row(t, d + 1, T, D))
            idx = (torch.stack((left, right), -1) + batch_off.unsqueeze(-1)).reshape(-1)
            child_k = GatherRows.apply(flat_kv, idx).view(B, T, K, 2, 2 * Dh)[..., :Dh].to(ft)
            content = ((child_k[..., 1, :] - child_k[..., 0, :]) * qr_k).sum(-1)   # [B, T, K]
            feats = torch.stack((
                log_len,
                torch.full_like(log_len, d),
                torch.full_like(log_len, float(level)),
                torch.log2((t - mid + 1).clamp_min(1).to(ft)).expand(B, T, K),
            ), -1).unsqueeze(2)                                                  # [B, T, 1, K, F]
            z = self.head.logits(u, feats).squeeze(2) + gain * content           # [B, T, K]
            with torch.no_grad():
                bit = torch.bernoulli(torch.sigmoid(z)).bool() & decide
            sign = bit.to(z.dtype) * 2 - 1
            lp_keep = F.logsigmoid(sign * z)
            lp_flip = F.logsigmoid(-sign * z)
            sibling = torch.where(bit, left, right)
            rows.append(torch.where(decide, sibling, torch.full_like(sibling, -1)))
            lps.append(path_lp + lp_flip)
            path_lp = torch.where(decide, path_lp + lp_keep, path_lp)
            lo = torch.where(bit, mid, lo)
        rows.append(lo)                                                          # leaf token
        lps.append(path_lp)
        rows = torch.stack(rows, -1).reshape(B, T, K * (D + 1))
        lps = torch.stack(lps, -1).reshape(B, T, K * (D + 1))
        return rows, lps

    def attend_gather(self, q: Tensor, table: Tensor, rows: Tensor, score_bias: Tensor) -> Tensor:
        """Reference: explicit row gather. ``q`` ``[B, T, H, Dh]``; ``rows``/
        ``score_bias`` ``[B, T, P]`` (``-1`` = masked)."""
        B, T, H, Dh = q.shape
        R = table.size(1)
        valid = rows >= 0
        idx = (rows.clamp_min(0) + (torch.arange(B, device=q.device) * R).view(B, 1, 1)).reshape(-1)
        cand = GatherRows.apply(table.reshape(B * R, 2 * Dh), idx).view(B, T, -1, 2 * Dh)
        k_c, v_c = cand.split(Dh, -1)
        sd = torch.promote_types(q.dtype, torch.float32)
        s = torch.einsum("bthd,btpd->bthp", q, k_c).to(sd) * SCORE_SCALE + score_bias.unsqueeze(2).to(sd)
        s = s.masked_fill(~valid.unsqueeze(2), MASKED_SCORE)
        w = torch.softmax(s, -1) * valid.unsqueeze(2).any(-1, keepdim=True)
        return torch.einsum("bthp,btpd->bthd", w.type_as(v_c), v_c)

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.size(0), x.size(1)
        if T != self.seq_len:
            raise ValueError(f"TreePointerAttention built for seq_len {self.seq_len}, got {T}")
        q, k, v, qr = self.qkv(x)
        table, log_count = build_row_table(k, v)
        # the descent's per-depth child gathers ([B, T, K, 2, Dh] x D) are
        # recomputed in backward; the checkpoint replays the RNG so the
        # Bernoulli draws (and hence the rows) are identical
        rows, log_prior = (checkpoint(self.route, x, qr, table, use_reentrant=False)
                           if torch.is_grad_enabled() else self.route(x, qr, table))
        score_bias = log_prior + log_count[rows.clamp_min(0)]
        if self.local_window:
            # token rows t - w, w in [0, W): real keys, prior mass 1; a token
            # also reached as a leaf appears twice and adds its mass, as in iptr
            local = (torch.arange(T, device=x.device).view(1, T, 1)
                     - torch.arange(self.local_window, device=x.device)).expand(B, T, -1)
            rows = torch.cat((rows, torch.where(local >= 0, local, torch.full_like(local, -1))), -1)
            score_bias = torch.cat((score_bias, torch.zeros_like(local, dtype=score_bias.dtype)), -1)
        if self.backend == "fused":
            P = rows.size(-1)
            pad = max(16, 1 << (P - 1).bit_length()) - P
            if pad:
                rows = F.pad(rows, (0, pad), value=-1)
                score_bias = F.pad(score_bias, (0, pad))
            y = fused_rows_attend(q, table, rows, score_bias)
        else:
            y = self.attend_gather(q, table, rows, score_bias)
        return self.proj(y.reshape(B, T, self.num_heads * self.head_dim))


class TreeBlock(nn.Module):
    def __init__(self, dim: int, seq_len: int, num_pointers: int, width: int,
                 mlp_hidden: "int | None" = None, head_dim: int = 128, backend: str = "fused",
                 local_window: int = 0):
        super().__init__()
        self.attn = TreePointerAttention(dim, seq_len, num_pointers, width, head_dim, backend, local_window)
        self.mlp = MLP(dim, mlp_hidden)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TreeGPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, seq_len: int,
                 num_pointers: int, width: int, mlp_hidden: "int | None" = None,
                 head_dim: int = 128, backend: str = "fused", local_window: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            TreeBlock(model_dim, seq_len, num_pointers, width, mlp_hidden, head_dim, backend, local_window)
            for _ in range(num_layers)
        ])
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

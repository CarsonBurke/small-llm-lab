"""Chunk-parallel CWM core: the lagged contracts (``CWMConfig.lag``) computed
round by round over all tokens of a chunk at once.

Under lag, no read of a token depends on a later round of an earlier token
of its own chunk: feedback reads (sources above the reader's round) see the
chunk-start state, and legal reads (sources up to the reader's round) see
writes of rounds the chunk has already computed for every token. So a chunk
is a stack of rounds, each one a batch over ``batch x lag_chunk`` rows, and
only the chunks run in sequence.

* **Banks, in WY form.** The chunk's delta-rule commits of one source,
  ``M_j = M_{j-1} D_j + u_j k_j^T`` with ``D_j = diag(exp(g_j))`` on the key
  dim and ``u_j = beta_j (v_j - M_{j-1} D_j k_j)``, unroll from the
  chunk-start state ``M0`` to ``M_j = M0 G_j + sum_{i<=j} u_i k_i^T
  G_j / G_i`` (``G_j`` the cumulative decay). The ``u`` solve one unit
  lower-triangular ``[C, C]`` system; a read of the state before token j
  and the chunk-end state are then products with ``M0`` and ``u``. Decay
  ratios are factored through a per-channel reference inside the chunk, so
  every factor stays within ``exp(+-5 C / 2)``: plain fp32 for ``C <= 32``.
  A source's writes and a round's reads are one Triton program per bank
  tile each (``chunk_kernels.wy_write``, ``wy_read``), the ``[C, C]``
  products in registers.
* **Store.** A round's legal reads attend, per token, over the previous
  ``store_window`` tokens' entries of its legal sources: the window kept from
  earlier chunks plus the chunk's own entries, which the round has
  already computed. Feedback reads see only the kept window. Both reads
  of a round are one Triton program per (row, head) over the whole window
  (``chunk_kernels.store_attend``).
* **Prepared weights.** A chunk runs on ``fused.prepare``'s weights (gains
  folded, every projection of one input concatenated): per round one GEMM
  for all reads and writes of its input, one for the read outputs, and one
  per entry for its depth keys and values for all later rounds. Each
  round reads every source's state once, and every bank-state product is
  a Triton kernel that streams the state once (``chunk_kernels.py``).

Two executors share ``chunk_forward``:

* ``ParallelCore``: autograd over the chunk loop (CPU tests, reference).
* ``GraphedParallelCore``: training and evaluation on CUDA graphs, laid out
  like ``graphed.GraphedCore``. The compiled chunk is captured twice: a
  forward graph, replayed once per chunk while the sweep saves each
  chunk's entry bank state (the boundary checkpoints) and store entries,
  and a backward graph that recomputes the chunk from its boundary and
  backpropagates the output gradient and the carried bank and store
  gradients through it under autograd, replayed from the last chunk to
  the first.

``pretraining/tests/test_cwm_parallel.py`` holds both to the module path
(``CWM.core``) in values and gradients.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from pretraining.cwm.chunk_kernels import bank_accumulate, bank_end, bank_read, store_attend, wy_read, wy_write
from pretraining.cwm.fused import FP32_PREPARED, PREPARED, Layout, prepare
from pretraining.cwm.model import (
    CWM,
    DECAY_LOWER_BOUND,
    CWMConfig,
    gated_head_norm,
    l2norm,
    placebo_shift,
    rms,
    rope,
    sink_attention,
)

# exp(-DECAY_LOWER_BOUND * C / 2) must stay well inside fp32.
MAX_LAG_CHUNK = 32


def supports(model: CWM) -> bool:
    """Whether the model's function is chunk-parallel: lagged, or exact
    without feedback (the lattice, or feedback ablated)."""
    return model.cfg.lag != "none" or model.feedback is None


def check_supported(model: CWM) -> None:
    if not supports(model):
        raise ValueError("an exact core with feedback is token-sequential (fused.py)")
    if model.cfg.bank_dtype != "float32":
        # The chunk core never rounds the state; the module rounds it per token.
        raise ValueError("the chunk-parallel core keeps fp32 banks")
    if model.cfg.lag_chunk > MAX_LAG_CHUNK:
        raise ValueError(f"lag_chunk above {MAX_LAG_CHUNK} needs sub-chunked decay ratios")


# ---------------------------------------------------------------------------
# Banks
# ---------------------------------------------------------------------------


class ChunkBank:
    """One chunk of delta-rule writes in WY form, for any leading batch dims
    ``*`` (rows, heads, sources), against the chunk-start state ``M0 [*, Dv,
    Dk]`` fp32. ``key``, ``log_decay [*, C, Dk]``, ``value [*, C, Dv]``,
    ``beta [*, C]``. The state itself is not a field: ``stack`` takes it as
    one tensor, so stacking sources never copies it."""

    FIELDS = ("key", "log_decay", "U")

    def __init__(self, M0: Tensor, key: Tensor, value: Tensor, beta: Tensor, log_decay: Tensor):
        self.M0, self.key, self.log_decay = M0, key, log_decay
        self.U = wy_write(M0, key, value, beta, log_decay)

    @classmethod
    def stack(cls, banks: list[ChunkBank], M0: Tensor) -> ChunkBank:
        """Independent banks as one, along a new leading dim; ``M0`` their
        chunk-start states, stacked the same way."""
        out = cls.__new__(cls)
        out.M0 = M0
        for f in cls.FIELDS:
            setattr(out, f, torch.stack([getattr(b, f) for b in banks]))
        return out

    def read(self, q: Tensor) -> Tensor:
        """For stacked banks ``[s, *]``: ``q [*, C, Dk]`` -> ``M_{j-1} q_j
        [s, *, C, Dv]`` per bank: token j reads the state its commit has not
        yet changed."""
        return wy_read(q, self.M0, self.key, self.log_decay, self.U)

    def end(self) -> Tensor:
        """The state after the chunk's last commit, ``[*, Dv, Dk]``."""
        G = self.log_decay.cumsum(-2)
        last = G[..., -1:, :]
        return bank_end(self.M0, last.squeeze(-2).exp(), self.U, self.key * (last - G).exp())


# ---------------------------------------------------------------------------
# One chunk, on prepared weights
# ---------------------------------------------------------------------------


def prepared_weights(model: CWM, dtype: torch.dtype) -> dict[str, Tensor]:
    """``fused.prepare`` with the GEMM weights in the activation dtype, as
    the chunk reads them (differentiable)."""
    return {n: t if n in FP32_PREPARED else t.to(dtype) for n, t in prepare(model).items()}


def chunk_forward(
    cfg: CWMConfig,
    freq: Tensor,
    w: dict[str, Tensor],
    u: Tensor,
    banks: Tensor,
    window_keys: Tensor | None,
    window_values: Tensor | None,
    start: Tensor,
    feedback: str | None,
    taps: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, tuple[Tensor, Tensor, Tensor, Tensor] | None]:
    """One chunk on the prepared weights ``w`` (``prepared_weights``):
    ``u [B, C, d]``, the chunk-start ``banks [S, B, H, Dv, Dk]`` fp32, the
    store entries of the ``W`` tokens before the chunk (``window_* [B, W,
    S, Hs, Ds]``) and the chunk's first position ``start`` (a device
    scalar). Returns the chunk's outputs ``[B, C, d]``, the banks after it,
    its own store entries ``[B, C, S, Hs, Ds]`` and, with ``taps``, the
    operands of its deferred state gradient.

    The layout is the fused executor's (``fused.Layout``): a round's reads
    and writes are one GEMM of its normalized input (gains folded), and an
    entry's depth keys and values for every later round are one GEMM.

    ``taps`` (zeros: ``[R, S, B, H, C, Dv]`` added to every round's reads
    of every source, ``[S, B, H, C, Dv]`` to the written values) defer the
    gradient of ``banks``: the reads and writes then see the state
    detached, so autograd leaves ``banks`` only the chunk-end term, and
    the taps' gradients, with the returned queries ``[R, B, H, C, Dk]``
    (legal, feedback) and written keys and log decays ``[S, B, H, C,
    Dk]``, give the rest in one pass (``chunk_kernels.bank_accumulate``)."""
    lay = Layout(cfg)
    B, C, d = u.shape
    CB = B * C
    R, S, H, D = cfg.rounds, cfg.sources, cfg.bank_heads, cfg.bank_dim
    Hd, Dd, hd = cfg.depth_heads, cfg.depth_dim, lay.hd
    Hs, Ds = cfg.store_heads, cfg.store_dim
    exact_legal = cfg.lag != "all"
    # The state the reads and writes see: detached when its gradient is deferred.
    state = banks if taps is None else banks.detach()
    shift = placebo_shift(B) if feedback == "placebo" else 0
    posf = (start + torch.arange(C, device=u.device)).float().repeat(B)  # rows are (b, c)
    lr = slice(lay.lr, lay.lr + 2 * cfg.low_rank)

    def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
        return F.linear(x, weight, bias)

    def heads(x: Tensor, n: int) -> Tensor:
        """Rows (b, c) -> ``[B, n, C, -1]``, contiguous: the bank ops take
        contiguous operands, and inductor writes this layout from the
        producing kernel instead of a separate copy."""
        return x.view(B, C, n, -1).transpose(1, 2).contiguous()

    def low_rank(i: int, P: Tensor) -> tuple[Tensor, Tensor]:
        """Source i's low-rank projections ``Z`` (decay logits, then the bank
        gate) and the decay bias. The per-head rate ``exp(A_log)`` is folded
        into the decay rows of the weight and into the bias, so its gradient
        comes from the weight's GEMM, not a reduction over every token."""
        rate = w["alog"][i].exp().repeat_interleave(D)  # [H * D]
        weight = w["Wlr"][i]
        weight = torch.cat([(weight[: lay.HD].float() * rate[:, None]).type_as(weight), weight[lay.HD :]])
        return linear(P[:, lr], weight), rate * w["dt"][i]

    def writes(i: int, P: Tensor) -> tuple[ChunkBank, Tensor, Tensor | None, Tensor | None]:
        """Source i's writes of the chunk: its WY bank and store entries,
        and its low-rank projections ``Z``."""
        Z, bias = low_rank(i, P)
        key = l2norm(P[:, lay.k : lay.k + lay.HD].view(CB, H, D).float())
        value = heads(P[:, lay.v : lay.v + lay.HD].float(), H)
        beta = cfg.beta_max * torch.sigmoid(P[:, lay.beta : lay.beta + H].float())
        log_decay = DECAY_LOWER_BOUND * torch.sigmoid(Z[:, : lay.HD].float() + bias).view(CB, H, D)
        if taps is not None:
            value = value + taps[1][i]
        bank = ChunkBank(state[i], heads(key, H), value, heads(beta, H).squeeze(-1), heads(log_decay, H))
        if not cfg.store:
            return bank, Z, None, None
        sk = rms(P[:, lay.sk : lay.sk + lay.HSD].view(CB, Hs, Ds)) + w["tag"][i].to(P.dtype)
        sk = rope(sk, posf, freq).view(B, C, Hs, Ds)
        return bank, Z, sk, P[:, lay.sv : lay.sv + lay.HSD].view(B, C, Hs, Ds)

    def depth_read(Y: list[Tensor], i: int) -> Tensor:
        """Round i + 1's read over entries 0..i (``Y[j]``: entry j's
        projections, ``fused.Layout.entry``)."""
        q = rms(Y[i][:, :hd].view(CB, Hd, Dd))
        ks, vs = [], []
        for j in range(i + 1):
            at = hd + 2 * hd * (i - j)  # entry j serves rounds j + 1..R
            ks.append(rms(Y[j][:, at : at + hd].view(CB, Hd, Dd)))
            vs.append(Y[j][:, at + hd : at + 2 * hd].view(CB, Hd, Dd))
        o = sink_attention(q, torch.stack(ks, 1), torch.stack(vs, 1), None, 0.12 * 128 / Dd)
        return o.reshape(CB, hd)

    def entry(j: int, e: Tensor) -> Tensor:
        at, width = lay.entry(j)
        return linear(e, w["WE"][at : at + width], w["bE"][at : at + width])

    if cfg.store:
        chunk_keys, chunk_values = [], []
    h = u.reshape(CB, d)
    Y = [entry(0, rms(h))]
    sources: list[ChunkBank] = []
    queries, fb_queries = [], []
    for i in range(R):
        r = i + 1
        h = h + linear(depth_read(Y, i).type_as(h), w["Wdo"][i], w["bdo"][i])
        P = linear(rms(h), w["Wx"][i], w["bx"][i])
        bank, Z, sk, sv = writes(i, P)
        sources.append(bank)
        # Legal sources read the state before each token (exact: start state
        # and the chunk's writes, ``wy_read``) or the chunk-start state (lag
        # "all"); feedback sources the chunk-start state, from row b - B/2
        # (placebo). Each round passes over every source's state once.
        q = heads(l2norm(P[:, lay.q : lay.q + lay.HD].view(CB, H, D).float()) * D**-0.5, H)  # [B, H, C, D]
        fb_q = q.roll(-shift, 0) if shift else q
        if exact_legal:
            legal = ChunkBank.stack(sources, state[:r]).read(q)  # [r, B, H, C, D]
        elif feedback and not shift:
            reads = bank_read(q, state)
            legal, fb_reads = reads[:r], reads[r:]
        else:
            legal = bank_read(q, state[:r])
        if feedback and (exact_legal or shift):
            fb_reads = bank_read(fb_q, state[r:])
        if taps is not None:
            queries.append(q)
            fb_queries.append(fb_q)
            legal = legal + taps[0][i, :r]
            if feedback:
                fb_reads = fb_reads + taps[0][i, r:]
        gate = Z[:, lay.HD : 2 * lay.HD].view(CB, H, D)

        def path(reads: Tensor, mix: Tensor, norm: Tensor) -> Tensor:
            """Mixed, normalized and gated reads ``[s, B, H, C, D]`` -> rows."""
            n = reads.size(0)
            reads = reads.permute(1, 3, 2, 4, 0).reshape(CB, H, D, n)
            mixed = reads[..., 0] if n == 1 else (reads * mix.view(CB, H, 1, n).float().softmax(-1)).sum(-1)
            return gated_head_norm(mixed, gate, norm).type_as(h).reshape(CB, lay.HD)

        RO = [path(legal, P[:, lay.mix : lay.mix + H * r], w["wn"][i, 0])]
        if feedback:
            RO.append(path(fb_reads.roll(shift, 1), P[:, lay.mix + H * r : lay.mix + H * S], w["wn"][i, 1]))
        else:
            RO.append(h.new_zeros(CB, lay.HD))
        if cfg.store:
            chunk_keys.append(sk)
            chunk_values.append(sv)
            sq = rope(rms(P[:, lay.sq : lay.sq + lay.HSD].view(CB, Hs, Ds)), posf, freq).view(B, C, Hs, Ds).float()
            chunk = (torch.stack(chunk_keys, dim=2), torch.stack(chunk_values, dim=2))
            out, fb_out, _, _ = store_attend(sq, window_keys, window_values, *chunk, start, cfg.store_scale, r,
                                             feedback is not None, shift)
            RO.append(out.type_as(h).reshape(CB, lay.HSD))
            RO.append(fb_out.type_as(h).reshape(CB, lay.HSD) if feedback else h.new_zeros(CB, lay.HSD))
        h = h + linear(torch.cat(RO, dim=1), w["Wout"][i], w["bout"][i])
        A = linear(rms(h), w["Wfc"][i], w["bfc"][i]).relu().square()
        h = h + linear(A, w["Wp"][i], w["bp"][i])
        if i < R - 1:
            Y.append(entry(i + 1, rms(h)))
    P = linear(rms(h), w["Wx"][R], w["bx"][R])
    bank, _, sk, sv = writes(R, P)
    sources.append(bank)
    # The chunk-end state sees ``banks`` itself: its term of the state
    # gradient stays with autograd.
    final = ChunkBank.stack(sources, banks)
    deferred = None
    if taps is not None:
        deferred = (torch.stack(queries), torch.stack(fb_queries), final.key, final.log_decay)
    keys = values = None
    if cfg.store:
        keys = torch.stack(chunk_keys + [sk], dim=2)
        values = torch.stack(chunk_values + [sv], dim=2)
    return h.view(B, C, d), final.end(), keys, values, deferred


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------


class ParallelCore:
    """The core of ``model`` for inputs ``u [B, T, d]`` under autograd,
    chunk by chunk. ``model.feedback`` is read per call (evaluation ablates
    it)."""

    def __init__(self, model: CWM):
        check_supported(model)
        self.model = model

    def __call__(self, u: Tensor) -> Tensor:
        model, cfg = self.model, self.model.cfg
        check_supported(model)  # feedback can be re-enabled between calls
        B, T, _ = u.shape
        C = cfg.lag_chunk
        if T % C:
            raise ValueError(f"sequence length {T} is not a multiple of lag_chunk {C}")
        banks, keys, values = model.initial_state(B, u.device, u.dtype)
        feedback = model.feedback
        weights = prepared_weights(model, u.dtype)
        outputs = []
        for c0 in range(0, T, C):
            start = torch.tensor(c0, device=u.device)
            h, banks, new_keys, new_values, _ = chunk_forward(
                cfg, model.store_freq, weights, u[:, c0 : c0 + C], banks, keys, values, start, feedback
            )
            if cfg.store:
                keys = torch.cat([keys, new_keys], dim=1)[:, -cfg.store_window :]
                values = torch.cat([values, new_values], dim=1)[:, -cfg.store_window :]
            outputs.append(h)
        return torch.cat(outputs, dim=1)


class GraphedParallelCore:
    """The core on CUDA graphs for one model, batch size, sequence length
    and feedback mode (``model.feedback`` is read at construction), with the
    interface of ``graphed.GraphedCore``. ``train=False`` keeps no gradient
    buffers or boundaries and captures only the forward graph. Off CUDA, or
    with ``graphs=False``, the same procedures run eagerly."""

    def __init__(self, model: CWM, batch: int, seq_len: int, train: bool, graphs: bool | None = None, compile: bool | None = None, warmup: int = 2):
        check_supported(model)
        cfg = model.cfg
        C = cfg.lag_chunk
        if seq_len % C:
            raise ValueError(f"sequence length {seq_len} is not a multiple of lag_chunk {C}")
        self.model, self.batch, self.seq_len, self.train = model, batch, seq_len, train
        self.feedback = model.feedback
        self.chunks = seq_len // C
        device = model.embed.weight.device
        self.dtype = model.embed.weight.dtype
        self.graphs = device.type == "cuda" if graphs is None else graphs
        self.fn = chunk_forward
        if self.graphs if compile is None else compile:
            self.fn = torch.compile(chunk_forward, fullgraph=True, dynamic=False)
        self.freq = model.store_freq.to(device, torch.float32)
        S, H, D, d = cfg.sources, cfg.bank_heads, cfg.bank_dim, cfg.dim
        B, W = batch, cfg.store_window

        def zeros(*shape, dtype=torch.float32) -> Tensor:
            return torch.zeros(shape, device=device, dtype=dtype)

        # Graph inputs and outputs.
        prepared = prepared_weights(model, self.dtype)
        self.weights = {n: torch.zeros_like(t, requires_grad=False) for n, t in prepared.items()}
        self.U = zeros(B, C, d, dtype=self.dtype)
        self.M = zeros(S, B, H, D, D)
        self.start = zeros(dtype=torch.long)
        self.H = zeros(B, C, d, dtype=self.dtype)
        if cfg.store:
            entry = (cfg.sources, cfg.store_heads, cfg.store_dim)
            self.WK, self.WV = zeros(B, W, *entry, dtype=self.dtype), zeros(B, W, *entry, dtype=self.dtype)
            self.CK, self.CV = zeros(B, C, *entry, dtype=self.dtype), zeros(B, C, *entry, dtype=self.dtype)
            # Every token's entries, after W empty slots: chunk c0's window
            # is [c0, c0 + W).
            self.EK, self.EV = zeros(B, W + seq_len, *entry, dtype=self.dtype), zeros(B, W + seq_len, *entry, dtype=self.dtype)
        if train:
            self.boundaries = zeros(self.chunks, S, B, H, D, D)
            self.dOut, self.dU = zeros(B, C, d, dtype=self.dtype), zeros(B, C, d, dtype=self.dtype)
            self.dM = zeros(S, B, H, D, D)
            # Zero taps of the deferred state gradient (``chunk_forward``).
            self.taps = (zeros(cfg.rounds, S, B, H, C, D), zeros(S, B, H, C, D))
            self.grads = {n: zeros(*t.shape) for n, t in prepared.items()}
            if cfg.store:
                self.dCK, self.dCV = zeros(B, C, *entry, dtype=self.dtype), zeros(B, C, *entry, dtype=self.dtype)
                self.dWK, self.dWV = zeros(B, W, *entry, dtype=self.dtype), zeros(B, W, *entry, dtype=self.dtype)
                # fp32: a window's gradient sums the reads of several chunks.
                self.dEK, self.dEV = zeros(B, W + seq_len, *entry), zeros(B, W + seq_len, *entry)
        self.pending = False
        if self.graphs:
            self._capture(warmup)

    @torch.no_grad()
    def load(self, prepared: dict[str, Tensor]) -> None:
        for name, buf in self.weights.items():
            buf.copy_(prepared[name])

    # -- the two graphed procedures -------------------------------------

    def _run(self, weights: dict[str, Tensor], u: Tensor, banks: Tensor, window: tuple[Tensor | None, Tensor | None],
             taps: tuple[Tensor, Tensor] | None = None):
        cfg = self.model.cfg
        return self.fn(cfg, self.freq, weights, u, banks, *window, self.start, self.feedback, taps)

    def _window(self) -> tuple[Tensor | None, Tensor | None]:
        return (self.WK, self.WV) if self.model.cfg.store else (None, None)

    def forward_chunk(self) -> None:
        h, banks, keys, values, _ = self._run(self.weights, self.U, self.M, self._window())
        self.H.copy_(h)
        self.M.copy_(banks)
        if self.model.cfg.store:
            self.CK.copy_(keys)
            self.CV.copy_(values)

    def backward_chunk(self) -> None:
        """Recompute the chunk from ``M`` (its boundary) and backpropagate
        ``dOut`` and the carried gradients of its end state (``dM``) and
        store entries (``dCK``, ``dCV``). Outputs: ``dU``, ``dM`` at the
        chunk's start, the window's gradient (``dWK``, ``dWV``), and weight
        gradients added to ``grads``. Autograd gives ``dM`` its chunk-end
        term; the reads' and writes' terms are deferred through the taps
        (``chunk_forward``) and added in one pass."""
        store = self.model.cfg.store

        def leaf(t: Tensor) -> Tensor:
            return t.detach().requires_grad_()

        with torch.enable_grad():
            weights = {n: leaf(t) for n, t in self.weights.items()}
            u, banks = leaf(self.U), leaf(self.M)
            taps = tuple(leaf(t) for t in self.taps)
            window = (leaf(self.WK), leaf(self.WV)) if store else (None, None)
            h, banks_out, keys, values, deferred = self._run(weights, u, banks, window, taps)
            outputs, d_outputs = [h, banks_out], [self.dOut, self.dM]
            inputs = [u, banks]
            if store:
                outputs += [keys, values]
                d_outputs += [self.dCK, self.dCV]
                inputs += window
            grads = torch.autograd.grad(outputs, inputs + list(taps) + list(weights.values()), d_outputs, allow_unused=True)
        targets = [self.dU, self.dM] + ([self.dWK, self.dWV] if store else [])
        for target, g in zip(targets, grads):
            if g is None:
                target.zero_()
            else:
                target.copy_(g)
        d_reads, d_values = grads[len(inputs) : len(inputs) + 2]
        q, q_fb, key, log_decay = (t.detach() for t in deferred)
        bank_accumulate(self.dM, q, q_fb, d_reads, key, log_decay, d_values, self.model.cfg.lag != "all")
        for buf, g in zip(self.grads.values(), grads[len(inputs) + 2 :]):
            if g is not None:
                buf.add_(g)

    def _capture(self, warmup: int) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            # Warmup compiles the chunk in both grad modes and lets cuBLAS
            # pick its kernels before capture.
            for _ in range(warmup):
                self.forward_chunk()
                if self.train:
                    self.backward_chunk()
        torch.cuda.current_stream().wait_stream(side)
        pool = torch.cuda.graph_pool_handle()
        with torch.no_grad():
            self.forward_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.forward_graph, pool=pool):
                self.forward_chunk()
            if self.train:
                self.backward_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.backward_graph, pool=pool):
                    self.backward_chunk()

    # -- sweeps -----------------------------------------------------------

    def _check(self, u: Tensor) -> None:
        if u.shape != (self.batch, self.seq_len, self.model.cfg.dim) or u.dtype != self.dtype:
            raise ValueError(f"the core was built for {(self.batch, self.seq_len)} {self.dtype}, got {tuple(u.shape)} {u.dtype}")
        if self.model.feedback != self.feedback:
            raise RuntimeError("model feedback mode differs from the one this core was built for")

    def _load_chunk(self, u: Tensor, c: int) -> slice:
        C, W = self.model.cfg.lag_chunk, self.model.cfg.store_window
        span = slice(c * C, (c + 1) * C)
        self.U.copy_(u[:, span])
        self.start.fill_(c * C)
        if self.model.cfg.store:
            self.WK.copy_(self.EK[:, c * C : c * C + W])
            self.WV.copy_(self.EV[:, c * C : c * C + W])
        return span

    @torch.no_grad()
    def forward(self, u: Tensor, save: bool) -> Tensor:
        """Core outputs ``[B, T, d]`` with the weights last ``load``ed."""
        self._check(u)
        if save and not self.train:
            raise RuntimeError("an evaluation core keeps no boundaries")
        if save and self.pending:
            raise RuntimeError("the previous training forward was never backpropagated")
        W = self.model.cfg.store_window
        out = torch.empty_like(u)
        self.M.zero_()
        if self.model.cfg.store:
            self.EK[:, :W].zero_()
            self.EV[:, :W].zero_()
        for c in range(self.chunks):
            if save:
                self.boundaries[c].copy_(self.M)
            span = self._load_chunk(u, c)
            if self.graphs:
                self.forward_graph.replay()
            else:
                self.forward_chunk()
            out[:, span].copy_(self.H)
            if self.model.cfg.store:
                self.EK[:, W + span.start : W + span.stop].copy_(self.CK)
                self.EV[:, W + span.start : W + span.stop].copy_(self.CV)
        self.pending = save
        return out

    @torch.no_grad()
    def backward(self, d_out: Tensor, u: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Gradients of the input and of every prepared weight (``PREPARED``
        order), for the last saved forward."""
        if not self.pending:
            raise RuntimeError("backward without a saved training forward")
        store, W = self.model.cfg.store, self.model.cfg.store_window
        for g in self.grads.values():
            g.zero_()
        self.dM.zero_()
        if store:
            self.dEK.zero_()
            self.dEV.zero_()
        d_u = torch.empty_like(u)
        for c in reversed(range(self.chunks)):
            self.M.copy_(self.boundaries[c])
            span = self._load_chunk(u, c)
            self.dOut.copy_(d_out[:, span])
            if store:
                # Reads of this chunk's entries by later chunks.
                self.dCK.copy_(self.dEK[:, W + span.start : W + span.stop])
                self.dCV.copy_(self.dEV[:, W + span.start : W + span.stop])
            if self.graphs:
                self.backward_graph.replay()
            else:
                self.backward_chunk()
            d_u[:, span].copy_(self.dU)
            if store:
                self.dEK[:, span.start : span.start + W] += self.dWK
                self.dEV[:, span.start : span.start + W] += self.dWV
        self.pending = False
        # Copies: autograd may keep a returned gradient as a parameter's
        # ``.grad``, and these buffers are reused by the next backward.
        return d_u, [self.grads[name].clone() for name in PREPARED]


class _CoreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: Tensor, core: GraphedParallelCore, *prepared: Tensor) -> Tensor:
        core.load(dict(zip(PREPARED, prepared)))
        ctx.core = core
        ctx.save_for_backward(u)
        return core.forward(u, save=True)

    @staticmethod
    def backward(ctx, d_out: Tensor):
        (u,) = ctx.saved_tensors
        d_u, grads = ctx.core.backward(d_out.contiguous(), u)
        return (d_u, None, *grads)


def graphed_parallel_core(core: GraphedParallelCore, u: Tensor) -> Tensor:
    """Differentiable core output ``[B, T, d]``: gradients reach ``u`` and,
    through ``fused.prepare``, the model's core parameters."""
    prepared = prepare(core.model)
    if not torch.is_grad_enabled():
        core.load(prepared)
        return core.forward(u, save=False)
    return _CoreFunction.apply(u, core, *prepared.values())

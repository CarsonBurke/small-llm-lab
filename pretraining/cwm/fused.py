"""Fused executor of the CWM core: the same function as ``CWM.step``, run
with a hand-written backward.

Why: the autograd step is bound by kernel count, not arithmetic -- about
2,100 kernels per token step (per-use weight casts, their gradient adds,
small separate projections) at a few microseconds each. This executor runs
each token step as 12 kernels per round in each direction (one per GEMM,
norm, depth read and combined bank + store read) plus one commit, and makes
three structural changes, none of which alters the function:

* **Prepared weights.** ``prepare`` builds, once per microbatch and
  differentiably, activation-dtype weight blocks with every RMSNorm gain
  folded in and every projection of one input concatenated (all of a
  round's reads and writes from ``in_norm(h)`` are one GEMM; the depth keys
  and values of an entry for all later rounds are one GEMM). Their
  gradients flow back to the module parameters through autograd.
* **Deferred weight gradients.** The backward step computes only input
  gradients. Each GEMM's inputs and output gradients stay in chunk stashes,
  and weight and bias gradients are one GEMM / reduction per chunk over
  ``chunk x batch`` rows instead of one per token. Per-row parameter
  partials (norm weights, decay rates, store tags) are reduced per chunk.
* **State in kernels.** Bank reads, the delta-rule commit, the windowed
  store and the depth read are Triton kernels (``kernels.py``). The bank
  state gradient of a token's reads is kept as rank-one terms and folded
  into the previous token's commit backward, so the fp32 state is streamed
  once per pass. Store entries live in a ``[B, T, ...]`` array indexed by
  token (their gradients in a ring of W token slots), so the store needs no
  boundary checkpoints.

Execution is chunked: the forward sweep saves the bank state at chunk starts
and every token's reads, the only activations that depend on other tokens.
The backward sweep rebuilds each chunk -- every round from the chunk's input
and the saved reads, in batch over ``chunk x batch`` rows with the per-token
kernels and GEMM configs (bitwise the forward's values), then the bank
states by re-running the commits from the boundary -- and walks its tokens
in reverse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
from torch import Tensor

from pretraining.cwm import kernels as K
from pretraining.cwm.model import CWM, CWMConfig, placebo_shift

FEEDBACK_MODES = {None: 0, "full": 1, "placebo": 2}
# Launch widths of the bank-state kernels, measured on an RTX 5090 at batch
# 128 (default config, core tokens/s): combined reads 2 warps (1: -1%,
# 4: -4%, 8: -13%); commits 2 (4: -3%, 8: -5%). Programmatic dependent
# launch of every kernel measured no gain (within 0.5%), so launches are
# plain stream-ordered.
READ_WARPS = 2
COMMIT_WARPS = 2
COMMIT_BACKWARD_WARPS = 2
# ``kernels.linear_kernel`` launch configs (BM, BN, BK, warps, stages,
# split-K) per per-token map; "d" prefixes the input-gradient map of the
# same weight. Calls on chunk-wide rows (the backward's rebuild) reuse the
# token map's config, so each row is computed exactly as in the forward.
# Swept on an RTX 5090 at 128 rows with weights streamed from DRAM
# (1.3-5.5 us each; cuBLAS took 1.7-25 us on the same shapes). The
# depth-entry maps narrow with j: WE0-2 (1,408-896 wide) take WE1's config,
# WE3-4 WE4's. At 128 rows a long-K map fills a fraction of the SMs, so
# the products only a norm consumes (``SPLIT_SITES``) may split K into
# fp32 partials that the norm kernel sums (Wp 5.9 -> 3.8 us, dWfc 5.4 ->
# 3.7, dWx 4.3 -> 3.3, dWE1-3 3.8-2.7 -> 3.1-2.4, Wout 2.9 -> 2.5).
_WIDE, _NARROW = (32, 32, 32, 4, 4, 1), (16, 32, 128, 4, 4, 1)
LINEAR_CONFIGS: dict[str, tuple[int, int, int, int, int, int]] = {
    "Wdo": (16, 32, 32, 4, 4, 1),
    "Wx": (32, 32, 64, 4, 3, 1),
    "Wlr": (16, 32, 32, 4, 4, 1),
    "Wout": (32, 32, 64, 4, 4, 2),
    "Wfc": (32, 64, 64, 4, 4, 1),
    "Wp": (32, 64, 64, 4, 3, 8),
    "WE0": _WIDE, "WE1": _WIDE, "WE2": _WIDE, "WE3": _NARROW, "WE4": _NARROW,
    "dWp": (64, 32, 64, 4, 4, 1),
    "dWfc": (32, 64, 64, 4, 3, 8),
    "dWout": (32, 32, 64, 4, 4, 1),
    "dWlr": (16, 32, 128, 4, 4, 1),
    "dWx": (32, 64, 64, 4, 4, 4),
    "dWdo": (16, 32, 128, 4, 3, 1),
    "dWE1": (32, 64, 64, 4, 4, 4),
    "dWE2": (32, 32, 64, 4, 4, 2), "dWE3": (32, 32, 64, 4, 4, 2), "dWE4": (32, 32, 64, 4, 4, 2),
}
DEFAULT_LINEAR_CONFIG = (16, 32, 128, 4, 4, 1)  # depth maps beyond WE4
SPLIT_SITES = ("Wout", "Wp", "dWfc", "dWx", "dWE1", "dWE2", "dWE3", "dWE4")
assert all(c[5] == 1 for site, c in LINEAR_CONFIGS.items() if site not in SPLIT_SITES)
RELU_SQUARE = K.LINEAR_RELU_SQUARE.value
RELU_SQUARE_BACKWARD = K.LINEAR_RELU_SQUARE_BACKWARD.value


def linear(site: str, a: Tensor, w: Tensor, out: Tensor, bias: Tensor | None = None, epilogue: int = 0,
           aux: Tensor | None = None, transposed: bool = False) -> None:
    """``out = epilogue(a @ w^T + bias)``, or ``a @ w`` when ``transposed``
    (``w`` a forward weight ``[N_out, N_in]``). Row-major operands with
    unit column stride; any row strides."""
    BM, BN, BK, warps, stages, split = LINEAR_CONFIGS.get(site, DEFAULT_LINEAR_CONFIG)
    M, k = a.shape
    if split > 1:
        # fp32 partials [split, M, n] (``FusedCore._product``), summed by
        # the consuming norm kernel.
        assert out.dim() == 3 and out.shape[0] == split and out.dtype == torch.float32
        assert bias is None and epilogue == 0
    else:
        out = out.unsqueeze(0)
    n = out.shape[2]
    assert a.stride(1) == 1 and out.stride(2) == 1 and out.shape[1] == M
    assert w.shape == ((k, n) if transposed else (n, k))
    swn, swk = (w.stride(1), w.stride(0)) if transposed else (w.stride(0), w.stride(1))
    kc = triton.cdiv(triton.cdiv(k, split), BK) * BK
    K.linear_kernel[(triton.cdiv(M, BM), triton.cdiv(n, BN), split)](
        a, w, out if bias is None else bias, out if aux is None else aux, out,
        M, n, k, kc, a.stride(0), swn, swk, out.stride(1) if aux is None else aux.stride(0), out.stride(1), out.stride(0),
        BM=BM, BN=BN, BK=BK, HAS_BIAS=bias is not None, EPILOGUE=epilogue, num_warps=warps, num_stages=stages,
    )


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@dataclass(frozen=True)
class Layout:
    """Column layouts shared by ``prepare``, the executor and the kernels."""

    cfg: CWMConfig

    def __post_init__(self):
        cfg = self.cfg
        for name in ("bank_dim", "store_dim", "depth_dim"):
            value = getattr(cfg, name)
            if value & (value - 1):
                raise ValueError(f"the fused core needs a power-of-two {name}, got {value}")
        if cfg.store and cfg.store_dim < 4:
            raise ValueError("store_dim must be at least 4")

    # Widths.
    @property
    def HD(self) -> int:
        return self.cfg.bank_heads * self.cfg.bank_dim

    @property
    def HSD(self) -> int:
        return self.cfg.store_heads * self.cfg.store_dim if self.cfg.store else 0

    @property
    def hd(self) -> int:
        return self.cfg.depth_heads * self.cfg.depth_dim

    # P: one source's projections of its input state.
    @property
    def k(self) -> int:
        return 0

    @property
    def v(self) -> int:
        return self.HD

    @property
    def beta(self) -> int:
        return 2 * self.HD

    @property
    def sk(self) -> int:
        return self.beta + self.cfg.bank_heads

    @property
    def sv(self) -> int:
        return self.sk + self.HSD

    @property
    def q(self) -> int:
        return self.sv + self.HSD

    @property
    def mix(self) -> int:
        return self.q + self.HD

    @property
    def sq(self) -> int:
        return self.mix + self.cfg.bank_heads * self.cfg.sources

    @property
    def lr(self) -> int:
        """The two low-rank inputs (decay, then gate), 16-byte aligned."""
        return _round_up(self.sq + self.HSD, 8)

    @property
    def nx(self) -> int:
        return _round_up(self.lr + 2 * self.cfg.low_rank, 8)

    # RO: one round's read outputs [bank legal | bank fb | store legal | store fb].
    @property
    def rw(self) -> int:
        return 2 * self.HD + 2 * self.HSD

    # Y: per entry j, [q_{j+1} | k_r, v_r for r = j+1..R].
    def entry(self, j: int) -> tuple[int, int]:
        R, hd = self.cfg.rounds, self.hd
        offset = hd * (j + 2 * R * j - j * (j - 1))
        return offset, hd * (1 + 2 * (R - j))

    @property
    def ny(self) -> int:
        offset, width = self.entry(self.cfg.rounds - 1)
        return offset + width


# ----------------------------------------------------------------------
# Prepared weights
# ----------------------------------------------------------------------

PREPARED = ("WE", "bE", "Wdo", "bdo", "Wx", "bx", "Wlr", "Wout", "bout", "Wfc", "bfc", "Wp", "bp", "wn", "alog", "dt", "tag")
FP32_PREPARED = ("wn", "alog", "dt", "tag")


def prepare(model: CWM) -> dict[str, Tensor]:
    """The executor's weights as fp32 functions of the module parameters
    (differentiable; gains folded, projections concatenated)."""
    cfg = model.cfg
    lay = Layout(cfg)
    R, S, H, d = cfg.rounds, cfg.sources, cfg.bank_heads, cfg.dim
    dev = model.embed.weight.device
    rounds = list(model.rounds)

    def f(t: Tensor) -> Tensor:
        return t.float()

    def zeros(*shape) -> Tensor:
        return torch.zeros(*shape, device=dev)

    WE, bE = [], []
    for j in range(R):
        blk = rounds[j]
        WE.append(f(blk.depth_q.weight) * f(blk.depth_norm.gains))
        bE.append(f(blk.depth_q.bias))
        for r in range(j + 1, R + 1):
            blk = rounds[r - 1]
            gain = f(blk.depth_norm.gains)
            WE += [f(blk.depth_k.weight) * gain, f(blk.depth_v.weight) * gain]
            bE += [f(blk.depth_k.bias) + f(blk.depth_tag[j]), f(blk.depth_v.bias)]

    def x_block(writer, blk, r: int | None, gain: Tensor) -> tuple[Tensor, Tensor]:
        rows, bias = [], []

        def add(weight: Tensor | None, b: Tensor | None, n: int, at: int):
            assert sum(w.size(0) for w in rows) == at, "layout mismatch"
            rows.append(f(weight) if weight is not None else zeros(n, d))
            bias.append(f(b) if b is not None else zeros(n))

        add(writer.bank_k.weight, None, lay.HD, lay.k)
        add(writer.bank_v.weight, None, lay.HD, lay.v)
        add(writer.bank_beta.weight, writer.bank_beta.bias, H, lay.beta)
        if cfg.store:
            add(writer.store_k.weight, writer.store_k.bias, lay.HSD, lay.sk)
            add(writer.store_v.weight, writer.store_v.bias, lay.HSD, lay.sv)
        if blk is None:
            add(None, None, lay.lr - lay.q, lay.q)
        else:
            add(blk.bank_q.weight, None, lay.HD, lay.q)
            for mix, n, at in (
                (blk.bank_mix_legal, H * r, lay.mix),
                (blk.bank_mix_fb, H * (S - r), lay.mix + H * r),
            ):
                add(None if mix is None else mix.weight, None if mix is None else mix.bias, n, at)
            if cfg.store:
                add(blk.store_q.weight, blk.store_q.bias, lay.HSD, lay.sq)
            add(None, None, lay.lr - lay.sq - lay.HSD, lay.sq + lay.HSD)
        add(writer.decay_a.weight, None, cfg.low_rank, lay.lr)
        add(None if blk is None else blk.bank_gate_a.weight, None, cfg.low_rank, lay.lr + cfg.low_rank)
        add(None, None, lay.nx - lay.lr - 2 * cfg.low_rank, lay.lr + 2 * cfg.low_rank)
        return torch.cat(rows) * gain, torch.cat(bias)

    Wx, bx, Wlr = [], [], []
    for i, blk in enumerate(rounds):
        w, b = x_block(blk.writer, blk, i + 1, f(blk.in_norm.gains))
        Wx.append(w)
        bx.append(b)
        Wlr.append(torch.block_diag(f(blk.writer.decay_b.weight), f(blk.bank_gate_b.weight)))
    w, b = x_block(model.out_writer, None, None, f(model.out_norm.gains))
    Wx.append(w)
    bx.append(b)
    Wlr.append(torch.block_diag(f(model.out_writer.decay_b.weight), zeros(lay.HD, cfg.low_rank)))

    def out_block(blk) -> tuple[Tensor, Tensor]:
        linears = [blk.bank_o_legal, blk.bank_o_fb] + ([blk.store_o_legal, blk.store_o_fb] if cfg.store else [])
        legal, fb = linears[0::2], linears[1::2]
        # Without feedback the module skips the feedback projections, biases
        # included; the kernels zero the feedback reads, so only the biases
        # need dropping.
        biased = legal + (fb if model.feedback is not None else [])
        return torch.cat([f(l.weight) for l in linears], dim=1), sum(f(l.bias) for l in biased)

    outs = [out_block(blk) for blk in rounds]
    writers = [blk.writer for blk in rounds] + [model.out_writer]
    prepared = {
        "WE": torch.cat(WE),
        "bE": torch.cat(bE),
        "Wdo": torch.stack([f(blk.depth_o.weight) for blk in rounds]),
        "bdo": torch.stack([f(blk.depth_o.bias) for blk in rounds]),
        "Wx": torch.stack(Wx),
        "bx": torch.stack(bx),
        "Wlr": torch.stack(Wlr),
        "Wout": torch.stack([w for w, _ in outs]),
        "bout": torch.stack([b for _, b in outs]),
        "Wfc": torch.stack([f(blk.mlp.fc.weight) * f(blk.mlp_norm.gains) for blk in rounds]),
        "bfc": torch.stack([f(blk.mlp.fc.bias) for blk in rounds]),
        "Wp": torch.stack([f(blk.mlp.proj.weight) for blk in rounds]),
        "bp": torch.stack([f(blk.mlp.proj.bias) for blk in rounds]),
        "wn": torch.stack([torch.stack([f(blk.bank_norm_legal), f(blk.bank_norm_fb)]) for blk in rounds]),
        "alog": torch.stack([f(w.A_log) for w in writers]),
        "dt": torch.stack([f(w.dt_bias) for w in writers]),
        "tag": torch.stack([f(w.store_tag) for w in writers]) if cfg.store else zeros(S, 1, 1),
    }
    assert tuple(prepared) == PREPARED
    return prepared


def _accumulate_mm(acc: Tensor, a: Tensor, b: Tensor) -> None:
    """``acc += a @ b`` with fp32 accumulation and output."""
    if a.dtype == acc.dtype:
        acc.addmm_(a, b)
    else:
        torch.addmm(acc, a, b, out_dtype=acc.dtype, out=acc)


# ----------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------


class FusedCore:
    """Buffers and chunk procedures of one (batch, seq_len, chunk, mode).

    The chunk procedures read only static buffers and the device scalar
    ``pos0`` (the chunk's first token), so they can be captured as CUDA
    graphs (``graphed.GraphedCore``) or run eagerly (CPU tests)."""

    def __init__(self, model: CWM, batch: int, seq_len: int, chunk: int, train: bool, dtype: torch.dtype):
        cfg = model.cfg
        if cfg.lag != "none":
            raise ValueError("the fused executor runs the exact contract (lag 'none'); lagged cores run in parallel.py")
        if seq_len % chunk:
            raise ValueError("seq_len must be a multiple of chunk")
        self.cfg, self.lay = cfg, Layout(cfg)
        self.batch, self.seq_len, self.chunk, self.train = batch, seq_len, chunk, train
        self.feedback = model.feedback
        self.fb = FEEDBACK_MODES[self.feedback]
        self.shift = placebo_shift(batch) if self.feedback == "placebo" else 0
        self.dtype = dtype
        dev = model.embed.weight.device
        self.device = dev
        self.freq = model.store_freq.to(dev, torch.float32)
        lay = self.lay
        R, S, H, D, d = cfg.rounds, cfg.sources, cfg.bank_heads, cfg.bank_dim, cfg.dim
        Fh, B, C = cfg.mlp_hidden, batch, chunk
        Hs, Ds, W = cfg.store_heads, cfg.store_dim, cfg.store_window
        self.WP = triton.next_power_of_2(W)
        # Bank and store heads share their launches (``kernels``, Launches).
        self.heads = H + (Hs if cfg.store else 0)

        def act(*shape):
            return torch.zeros(*shape, device=dev, dtype=dtype)

        def f32(*shape):
            return torch.zeros(*shape, device=dev, dtype=torch.float32)

        def bank(*shape):
            return torch.zeros(*shape, device=dev, dtype=cfg.bank_torch_dtype)

        shapes = self._prepared_shapes()
        self.weights = {n: (f32 if n in FP32_PREPARED else act)(*s) for n, s in shapes.items()}
        self.pos0 = torch.zeros(1, device=dev, dtype=torch.int32)
        # Stands in for the store arguments of a launch when there is no store.
        self.unused = f32(1)
        # Bank states (``bank_dtype``). The forward commits in place in slot
        # 0 (each commit program owns one [D, D] tile, and the token's reads
        # precede its commit); the backward rebuilds the state at every token
        # of the chunk (index t: before token t).
        self.M = bank(C + 1 if train else 1, S, B, H, D, D)
        if cfg.store:
            self.EK = act(B, seq_len, S, Hs, Ds)
            self.EV = act(B, seq_len, S, Hs, Ds)
        # Activations of the chunk (inputs of every GEMM and kernel).
        self.U = act(C, B, d)
        self.Y = act(C, B, lay.ny)
        self.E = act(R, C, B, d)  # E[j] = rms(log_j): u, h_1, ..., h_{R-1}
        self.OD = act(R, C, B, lay.hd)
        self.HA, self.HB, self.HC, self.MN = (act(R, C, B, d) for _ in range(4))
        self.XN = act(R + 1, C, B, d)
        self.P = act(R + 1, C, B, lay.nx)
        self.Z = act(R + 1, C, B, 2 * lay.HD)
        self.RO = act(R, C, B, lay.rw)
        self.MIXED = f32(R, C, 2, B, H, D)  # each read path's source mixture
        if cfg.store:
            self.STATS = f32(R, C, B, Hs, 2, 2)  # each store path's softmax (max, normalizer)
        self.F = act(R, C, B, Fh)
        self.A = act(R, C, B, Fh)
        self.t_d = act(B, d)
        # Split-K partials of the products only a norm consumes (``linear``),
        # over chunk rows for the backward's rebuild.
        splits = max(LINEAR_CONFIGS.get(site, DEFAULT_LINEAR_CONFIG)[5] for site in SPLIT_SITES)
        self.partials = f32(splits, C * B if train else B, d) if splits > 1 else self.unused
        if not train:
            return
        self.boundaries = bank(seq_len // chunk, S, B, H, D, D)
        # What the backward cannot rebuild in batch: the reads (they depend
        # on other tokens through the banks and the store) and their
        # mixtures and softmax statistics. Indexed by token; dim 1 is the
        # chunk dim of the matching chunk buffer.
        T = seq_len
        self.saved = {
            "RO": act(R, T, B, lay.rw),
            "MIXED": f32(R, T, 2, B, H, D),
        }
        if cfg.store:
            self.saved["STATS"] = f32(R, T, B, Hs, 2, 2)
        self.chunk_offsets = torch.arange(C, device=dev)
        self.tokens = torch.zeros(C, device=dev, dtype=torch.long)
        self.grads = {n: f32(*s) for n, s in shapes.items()}
        # The state gradient, stored like the state: bf16 banks get a bf16
        # G (computed in fp32, rounded once per token). Unlike the module's
        # fp32 straight-through gradient, but on a trained checkpoint it
        # moves the full gradient by 3e-3 relative, below the bf16 state's
        # own 8e-3 (NOTES.md 2026-10-01).
        self.G = bank(S, B, H, D, D)
        self.pend = f32(R, S, B, H, D)
        self.pend_q = f32(R, B, H, D)
        if cfg.store:
            # Entry gradients, in a ring of W token slots (``kernels``).
            self.dEK = f32(B, W, S, Hs, Ds)
            self.dEV = f32(B, W, S, Hs, Ds)
            self.sp = f32(R, B, Hs, S, self.WP)
            self.sd = f32(R, B, Hs, S, self.WP)
            self.sq = f32(R, B, Hs, Ds)
            self.sdo = f32(R, B, Hs, 2, Ds)
            self.sdelta = f32(R, B, Hs, 2)
            self.dtag_part = f32(C, S, B, Hs, Ds)
        self.dY = act(C, B, lay.ny)
        self.dHA, self.dHB, self.dHC = (act(R, C, B, d) for _ in range(3))
        self.dF = act(R, C, B, Fh)
        self.dP = act(R + 1, C, B, lay.nx)  # unwritten columns stay zero
        self.dZ = act(R + 1, C, B, 2 * lay.HD)
        self.dOut = act(C, B, d)
        self.dU = act(C, B, d)
        self.dwn_part = f32(C, R, B, H, 2, D)
        self.dalog_part = f32(C, S, B, H)
        self.ddt_part = f32(C, S, B, lay.HD)
        self.t_rw = act(B, lay.rw)
        self.t_hd = act(B, lay.hd)
        self.t_cd = act(C * B, d)

    def _prepared_shapes(self) -> dict[str, tuple[int, ...]]:
        cfg, lay = self.cfg, self.lay
        R, S, d, Fh = cfg.rounds, cfg.sources, cfg.dim, cfg.mlp_hidden
        return {
            "WE": (lay.ny, d),
            "bE": (lay.ny,),
            "Wdo": (R, d, lay.hd),
            "bdo": (R, d),
            "Wx": (R + 1, lay.nx, d),
            "bx": (R + 1, lay.nx),
            "Wlr": (R + 1, 2 * lay.HD, 2 * cfg.low_rank),
            "Wout": (R, d, lay.rw),
            "bout": (R, d),
            "Wfc": (R, Fh, d),
            "bfc": (R, Fh),
            "Wp": (R, d, Fh),
            "bp": (R, d),
            "wn": (R, 2, cfg.bank_dim),
            "alog": (S, cfg.bank_heads),
            "dt": (S, lay.HD),
            "tag": (S, cfg.store_heads, cfg.store_dim) if cfg.store else (S, 1, 1),
        }

    @torch.no_grad()
    def load(self, prepared: dict[str, Tensor]) -> None:
        for name, buf in self.weights.items():
            buf.copy_(prepared[name])

    # ------------------------------------------------------------------
    # Elementwise launches
    # ------------------------------------------------------------------
    def _product(self, site: str, out: Tensor) -> Tensor:
        """Where ``linear(site, ...)`` writes a product of ``out``'s shape
        that only a norm kernel consumes: ``out``, or for a split-K site its
        fp32 partials (one shared buffer)."""
        split = LINEAR_CONFIGS.get(site, DEFAULT_LINEAR_CONFIG)[5]
        return out if split == 1 else self.partials[:split, : out.shape[0], : out.shape[1]]

    @staticmethod
    def _splits(y: Tensor) -> tuple[int, int, Tensor]:
        """(SPLIT, partial stride, rows view) of a product or its partials."""
        return (1, 0, y) if y.dim() == 2 else (y.shape[0], y.stride(0), y[0])

    def _residual_rms(self, h: Tensor, y: Tensor | None, bias: Tensor | None, out_h: Tensor | None, out_n: Tensor) -> None:
        rows, n = out_n.shape
        add = y is not None
        split, stride_ys, y = self._splits(y if add else h)
        K.residual_rms_kernel[(rows,)](
            h, y, bias if add else h, out_h if add else h, out_n,
            h.stride(0), stride_ys, y.stride(0), (out_h if add else h).stride(0), out_n.stride(0),
            N=n, BLOCK=triton.next_power_of_2(n), ADD=add, SPLIT=split,
        )

    def _rms_backward(self, x: Tensor, dn: Tensor, dres: Tensor, dx: Tensor) -> None:
        rows, n = x.shape
        split, stride_dns, dn = self._splits(dn)
        K.rms_backward_kernel[(rows,)](
            x, dn, dres, dx, x.stride(0), stride_dns, dn.stride(0), dres.stride(0), dx.stride(0),
            N=n, BLOCK=triton.next_power_of_2(n), SPLIT=split,
        )

    @staticmethod
    def _flat_grid(n: int, block: int = 1024):
        return (triton.cdiv(n, block),)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _entry_inputs(self) -> None:
        """E_0 = rms(u) and entry 0's depth projections for the whole chunk."""
        cfg, lay, w = self.cfg, self.lay, self.weights
        C, B, d = self.chunk, self.batch, cfg.dim
        self._residual_rms(self.U.view(C * B, d), None, None, None, self.E[0].view(C * B, d))
        c0, n0 = lay.entry(0)
        linear("WE0", self.E[0].view(C * B, d), w["WE"][c0 : c0 + n0], self.Y.view(C * B, lay.ny)[:, c0 : c0 + n0], w["bE"][c0 : c0 + n0])

    def _commit(self, t: int, before: Tensor, after: Tensor, append: bool) -> None:
        """``after`` (which may be ``before``): every source's write of token t
        into ``before``; with ``append``, also token t's store entries."""
        cfg, lay, w = self.cfg, self.lay, self.weights
        append = append and cfg.store
        store = (w["tag"], self.EK, self.EV) if append else (self.unused,) * 3
        K.commit_forward_kernel[(cfg.sources, self.batch, self.heads if append else cfg.bank_heads)](
            before, after, self.P[0, t], self.Z[0, t], w["dt"], w["alog"], store[0], self.freq, self.pos0, *store[1:],
            self.batch, self.seq_len, t, self.P.stride(0), self.P.stride(2), self.Z.stride(0), self.Z.stride(2),
            lay.k, lay.v, lay.beta, cfg.beta_max, lay.sk, lay.sv,
            S=cfg.sources, H=cfg.bank_heads, D=cfg.bank_dim, HS=cfg.store_heads, DS=cfg.store_dim, STORE=append,
            num_warps=COMMIT_WARPS,
        )

    def _forward_step(self, t: int) -> None:
        cfg, lay, w = self.cfg, self.lay, self.weights
        R, S, B = cfg.rounds, cfg.sources, self.batch
        H, D = cfg.bank_heads, cfg.bank_dim
        Hs, Ds = cfg.store_heads, cfg.store_dim
        lr = slice(lay.lr, lay.lr + 2 * cfg.low_rank)
        y = self.Y[t]
        M = self.M[0]
        for i in range(R):
            h_in = self.U[t] if i == 0 else self.HC[i - 1, t]
            K.depth_forward_kernel[(B, cfg.depth_heads)](
                y, self.OD[i, t], y.stride(0), self.OD.stride(2), 0.12 * 128 / cfg.depth_dim,
                R=R, ROUND=i + 1, HD=lay.hd, DD=cfg.depth_dim,
            )
            linear("Wdo", self.OD[i, t], w["Wdo"][i], self.t_d)
            self._residual_rms(h_in, self.t_d, w["bdo"][i], self.HA[i, t], self.XN[i, t])
            linear("Wx", self.XN[i, t], w["Wx"][i], self.P[i, t], w["bx"][i])
            linear("Wlr", self.P[i, t, :, lr], w["Wlr"][i], self.Z[i, t])
            store = (self.EK, self.EV, self.STATS[i, t]) if cfg.store else (self.unused,) * 3
            K.read_forward_kernel[(B, self.heads)](
                M, self.P[i, t], self.Z[i, t], w["wn"][i], self.MIXED[i, t], self.RO[i, t],
                *store[:2], self.freq, self.pos0, store[2],
                B, self.seq_len, self.shift, t, self.P.stride(2), self.Z.stride(2), self.RO.stride(2),
                lay.q, lay.mix, lay.sq, 2 * lay.HD, D**-0.5, cfg.store_scale,
                S=S, H=H, D=D, HS=Hs, DS=Ds, W=cfg.store_window, WP=self.WP, ROUND=i + 1, FB=self.fb,
                SP=triton.next_power_of_2(S), STORE=cfg.store, num_warps=READ_WARPS,
            )
            prod = self._product("Wout", self.t_d)
            linear("Wout", self.RO[i, t], w["Wout"][i], prod)
            self._residual_rms(self.HA[i, t], prod, w["bout"][i], self.HB[i, t], self.MN[i, t])
            # A = relu(F)^2 directly; the backward rebuilds F.
            linear("Wfc", self.MN[i, t], w["Wfc"][i], self.A[i, t], w["bfc"][i], RELU_SQUARE)
            prod = self._product("Wp", self.t_d)
            linear("Wp", self.A[i, t], w["Wp"][i], prod)
            out_n = self.E[i + 1, t] if i < R - 1 else self.XN[R, t]
            self._residual_rms(self.HB[i, t], prod, w["bp"][i], self.HC[i, t], out_n)
            if i < R - 1:
                c, width = lay.entry(i + 1)
                linear(f"WE{i + 1}", self.E[i + 1, t], w["WE"][c : c + width], y[:, c : c + width], w["bE"][c : c + width])
        linear("Wx", self.XN[R, t], w["Wx"][R], self.P[R, t], w["bx"][R])
        linear("Wlr", self.P[R, t, :, lr], w["Wlr"][R], self.Z[R, t])
        self._commit(t, M, M, append=True)

    def forward_chunk(self) -> None:
        """Advance ``M[0]`` through the chunk (in place: ``M[0]`` is then the
        next chunk's entry), appending the chunk's store entries and
        (training) saving its activations. Output: ``HC[R - 1]``."""
        self._entry_inputs()
        for t in range(self.chunk):
            self._forward_step(t)
        if self.train:
            torch.add(self.chunk_offsets, self.pos0, out=self.tokens)
            for name, buf in self.saved.items():
                buf.index_copy_(1, self.tokens, self._chunk_buffer(name))

    def _chunk_buffer(self, name: str) -> Tensor:
        return getattr(self, name)

    def _rebuild(self) -> None:
        """The chunk's activations for the backward. Only the reads depend on
        other tokens: they are gathered from the forward's saves, and every
        round is recomputed from them and the chunk's input in batch over
        ``chunk x batch`` rows, with the per-token kernels and GEMM configs
        (so bitwise as the forward made it). The bank states follow by
        re-running the commits from the boundary."""
        cfg, lay, w = self.cfg, self.lay, self.weights
        R, C, B, d = cfg.rounds, self.chunk, self.batch, cfg.dim
        CB = C * B
        lr = slice(lay.lr, lay.lr + 2 * cfg.low_rank)
        torch.add(self.chunk_offsets, self.pos0, out=self.tokens)
        for name, buf in self.saved.items():
            torch.index_select(buf, 1, self.tokens, out=self._chunk_buffer(name))

        def rows(x: Tensor) -> Tensor:
            return x.view(CB, x.size(-1))

        Y, t = rows(self.Y), self.t_cd

        def projections(i: int) -> None:
            linear("Wx", rows(self.XN[i]), w["Wx"][i], rows(self.P[i]), w["bx"][i])
            linear("Wlr", rows(self.P[i])[:, lr], w["Wlr"][i], rows(self.Z[i]))

        self._residual_rms(rows(self.U), None, None, None, rows(self.E[0]))
        c0, n0 = lay.entry(0)
        linear("WE0", rows(self.E[0]), w["WE"][c0 : c0 + n0], Y[:, c0 : c0 + n0], w["bE"][c0 : c0 + n0])
        for i in range(R):
            h_in = self.U if i == 0 else self.HC[i - 1]
            K.depth_forward_kernel[(CB, cfg.depth_heads)](
                Y, rows(self.OD[i]), Y.stride(0), lay.hd, 0.12 * 128 / cfg.depth_dim,
                R=R, ROUND=i + 1, HD=lay.hd, DD=cfg.depth_dim,
            )
            linear("Wdo", rows(self.OD[i]), w["Wdo"][i], t)
            self._residual_rms(rows(h_in), t, w["bdo"][i], rows(self.HA[i]), rows(self.XN[i]))
            projections(i)
            prod = self._product("Wout", t)
            linear("Wout", rows(self.RO[i]), w["Wout"][i], prod)
            self._residual_rms(rows(self.HA[i]), prod, w["bout"][i], rows(self.HB[i]), rows(self.MN[i]))
            # F for the backward; A = relu(F)^2 as the forward's epilogue
            # computed it (from F rounded to the activation dtype).
            linear("Wfc", rows(self.MN[i]), w["Wfc"][i], rows(self.F[i]), w["bfc"][i])
            n = self.F[i].numel()
            K.relu_square_kernel[self._flat_grid(n)](self.F[i], self.A[i], n, BLOCK=1024)
            prod = self._product("Wp", t)
            linear("Wp", rows(self.A[i]), w["Wp"][i], prod)
            out_n = self.E[i + 1] if i < R - 1 else self.XN[R]
            self._residual_rms(rows(self.HB[i]), prod, w["bp"][i], rows(self.HC[i]), rows(out_n))
            if i < R - 1:
                c, width = lay.entry(i + 1)
                linear(f"WE{i + 1}", rows(self.E[i + 1]), w["WE"][c : c + width], Y[:, c : c + width], w["bE"][c : c + width])
        projections(R)
        for t_ in range(C):
            self._commit(t_, self.M[t_], self.M[t_ + 1], append=False)

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def _backward_step(self, t: int) -> None:
        cfg, lay, w = self.cfg, self.lay, self.weights
        R, S, B = cfg.rounds, cfg.sources, self.batch
        H, D = cfg.bank_heads, cfg.bank_dim
        Hs, Ds = cfg.store_heads, cfg.store_dim
        lr = slice(lay.lr, lay.lr + 2 * cfg.low_rank)
        P, dP, Z, dZ = self.P, self.dP, self.Z, self.dZ
        store = (self.dEK, self.dEV, self.dtag_part[t]) if cfg.store else (self.unused,) * 3
        K.commit_backward_kernel[(S, B, self.heads)](
            self.M[t], self.G, self.pend, self.pend_q, P[0, t], Z[0, t], w["dt"], w["alog"],
            dP[0, t], dZ[0, t], self.dalog_part[t], self.ddt_part[t], self.freq, self.pos0, *store,
            B, self.shift, t, P.stride(0), P.stride(2), Z.stride(0), Z.stride(2),
            dP.stride(0), dP.stride(2), dZ.stride(0), dZ.stride(2),
            lay.k, lay.v, lay.beta, cfg.beta_max, lay.sk, lay.sv,
            R=R, H=H, D=D, HS=Hs, DS=Ds, W=cfg.store_window, FB=self.fb, STORE=cfg.store,
            num_warps=COMMIT_BACKWARD_WARPS,
        )
        linear("dWlr", dZ[R, t], w["Wlr"][R], dP[R, t, :, lr], transposed=True)
        dn = self._product("dWx", self.t_d)
        linear("dWx", dP[R, t], w["Wx"][R], dn, transposed=True)
        self._rms_backward(self.HC[R - 1, t], dn, self.dOut[t], self.dHC[R - 1, t])
        dy = self.dY[t]
        for i in reversed(range(R)):
            if i < R - 1:
                # log_{i+1} fed round i+2's residual and entry i+1's depth
                # projections (complete: every later round is done).
                c, width = lay.entry(i + 1)
                dn = self._product(f"dWE{i + 1}", self.t_d)
                linear(f"dWE{i + 1}", dy[:, c : c + width], w["WE"][c : c + width], dn, transposed=True)
                self._rms_backward(self.HC[i, t], dn, self.dHA[i + 1, t], self.dHC[i, t])
            linear("dWp", self.dHC[i, t], w["Wp"][i], self.dF[i, t], epilogue=RELU_SQUARE_BACKWARD, aux=self.F[i, t], transposed=True)
            dn = self._product("dWfc", self.t_d)
            linear("dWfc", self.dF[i, t], w["Wfc"][i], dn, transposed=True)
            self._rms_backward(self.HB[i, t], dn, self.dHC[i, t], self.dHB[i, t])
            linear("dWout", self.dHB[i, t], w["Wout"][i], self.t_rw, transposed=True)
            if cfg.store:
                store = (self.EK, self.EV, self.STATS[i, t], self.sp[i], self.sd[i], self.sq[i], self.sdo[i], self.sdelta[i])
            else:
                store = (self.unused,) * 8
            K.read_backward_kernel[(B, self.heads)](
                self.M[t], P[i, t], Z[i, t], w["wn"][i], self.MIXED[i, t], self.t_rw,
                dP[i, t], dZ[i, t], self.dwn_part[t, i], self.pend[i], self.pend_q[i],
                *store[:2], self.freq, self.pos0, *store[2:],
                B, self.seq_len, self.shift, t, P.stride(2), Z.stride(2), self.t_rw.stride(0), dP.stride(2), dZ.stride(2),
                lay.q, lay.mix, lay.sq, 2 * lay.HD, D**-0.5, cfg.store_scale,
                S=S, H=H, D=D, HS=Hs, DS=Ds, W=cfg.store_window, WP=self.WP, ROUND=i + 1, FB=self.fb,
                SP=triton.next_power_of_2(S), STORE=cfg.store, num_warps=READ_WARPS,
            )
            linear("dWlr", dZ[i, t], w["Wlr"][i], dP[i, t, :, lr], transposed=True)
            dn = self._product("dWx", self.t_d)
            linear("dWx", dP[i, t], w["Wx"][i], dn, transposed=True)
            self._rms_backward(self.HA[i, t], dn, self.dHB[i, t], self.dHA[i, t])
            linear("dWdo", self.dHA[i, t], w["Wdo"][i], self.t_hd, transposed=True)
            K.depth_backward_kernel[(B, cfg.depth_heads)](
                self.Y[t], self.t_hd, dy, self.Y.stride(1), self.t_hd.stride(0), dy.stride(0), 0.12 * 128 / cfg.depth_dim,
                R=R, ROUND=i + 1, HD=lay.hd, DD=cfg.depth_dim,
            )
        if cfg.store:
            K.store_grad_kernel[(B, S, Hs)](
                self.dEK, self.dEV, self.sp, self.sd, self.sq, self.sdo, self.sdelta, self.pos0,
                B, self.shift, t, cfg.store_scale,
                R=R, S=S, HS=Hs, DS=Ds, W=cfg.store_window, WP=self.WP, FB=self.fb,
            )

    def _accumulate_chunk(self) -> None:
        """Input gradient of the chunk and every weight gradient, once."""
        cfg, lay, w, g = self.cfg, self.lay, self.weights, self.grads
        R, C, B, d = cfg.rounds, self.chunk, self.batch, cfg.dim
        CB = C * B
        lr = slice(lay.lr, lay.lr + 2 * cfg.low_rank)
        dY = self.dY.view(CB, lay.ny)
        c0, n0 = lay.entry(0)
        torch.mm(dY[:, c0 : c0 + n0], w["WE"][c0 : c0 + n0], out=self.t_cd)
        self._rms_backward(self.U.view(CB, d), self.t_cd, self.dHA[0].view(CB, d), self.dU.view(CB, d))
        for j in range(R):
            c, width = lay.entry(j)
            _accumulate_mm(g["WE"][c : c + width], dY[:, c : c + width].t(), self.E[j].view(CB, d))
        g["bE"] += dY.sum(0, dtype=torch.float32)
        for i in range(R):
            _accumulate_mm(g["Wdo"][i], self.dHA[i].view(CB, d).t(), self.OD[i].view(CB, -1))
            _accumulate_mm(g["Wout"][i], self.dHB[i].view(CB, d).t(), self.RO[i].view(CB, -1))
            _accumulate_mm(g["Wfc"][i], self.dF[i].view(CB, -1).t(), self.MN[i].view(CB, d))
            _accumulate_mm(g["Wp"][i], self.dHC[i].view(CB, d).t(), self.A[i].view(CB, -1))
        for i in range(R + 1):
            _accumulate_mm(g["Wx"][i], self.dP[i].view(CB, -1).t(), self.XN[i].view(CB, d))
            _accumulate_mm(g["Wlr"][i], self.dZ[i].view(CB, -1).t(), self.P[i].view(CB, -1)[:, lr])
        g["bdo"] += self.dHA.view(R, CB, d).sum(1, dtype=torch.float32)
        g["bout"] += self.dHB.view(R, CB, d).sum(1, dtype=torch.float32)
        g["bp"] += self.dHC.view(R, CB, d).sum(1, dtype=torch.float32)
        g["bfc"] += self.dF.view(R, CB, -1).sum(1, dtype=torch.float32)
        g["bx"] += self.dP.view(R + 1, CB, -1).sum(1, dtype=torch.float32)
        g["wn"] += self.dwn_part.sum((0, 2, 3))
        g["alog"] += self.dalog_part.sum((0, 2))
        g["dt"] += self.ddt_part.sum((0, 2))
        if cfg.store:
            g["tag"] += self.dtag_part.sum((0, 2))

    def backward_chunk(self) -> None:
        """Rebuild the chunk from ``M[0]`` (its boundary) and the saved
        activations, then propagate ``dOut`` and the carried state gradients
        (``G``, ``pend``, ``dEK``, ``dEV``) back through it. Outputs: ``dU``,
        the carried gradients at the chunk's entry, and weight gradients
        added to ``grads``."""
        self._rebuild()
        for t in reversed(range(self.chunk)):
            self._backward_step(t)
        self._accumulate_chunk()

    @torch.no_grad()
    def reset_backward(self) -> None:
        for buf in (self.G, self.pend, self.pend_q, *self.grads.values()):
            buf.zero_()
        if self.cfg.store:
            self.dEK.zero_()
            self.dEV.zero_()

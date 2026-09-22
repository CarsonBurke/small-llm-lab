"""Per-bank full-rank adapters over the packed pool: ``y[tile] = x[tile] @ W[bank(tile)]``.

The packed pool lays every bank's events out in whole 64-row tiles (the
kernel chunk), so each tile belongs to exactly one bank and a row's adapter is
a per-tile choice of weight matrix. That makes the adapter a static-shape
kernel: one program per (tile, output block) reads its bank index from a
device tensor and multiplies the tile by that bank's matrix. Nothing depends
on how many rows each bank received, so the operator compiles, CUDA-graphs
and costs the same as one dense matmul over the pool.

``torch._grouped_mm`` is not an option on this hardware: on sm_120 it runs the
fallback loop of one matmul per group with the offsets copied to the host,
which is several times slower than a dense matmul and cannot be captured.

The weight gradient sums ``x[tile]^T @ dy[tile]`` over each bank's tiles. The
layout keeps every bank's tiles contiguous (bank-major order), so a program
walks a runtime range of tiles; ``WEIGHT_SPLITS`` partial sums per bank keep
enough programs in flight and are reduced deterministically afterwards.
"""
from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl

from pretraining.nanogpt_mini.gated_delta_ops import CHUNK_SIZE, LIBRARY, TAGS, _require

TILE = CHUNK_SIZE
BLOCK_K = 64          # reduction step of the forward / input-gradient programs
WEIGHT_SPLITS = 8     # partial weight-gradient sums per bank
GRANULE = 64          # K and N are multiples of this; 128-wide blocks are used when they divide
Tensor2 = tuple[Tensor, Tensor]


@triton.jit
def _tile_matmul_kernel(x_ptr, w_ptr, bank_ptr, y_ptr,
                        stride_xr, stride_wg, stride_w1, stride_w2, stride_yr,
                        K: tl.constexpr, N: tl.constexpr, TRANSPOSED: tl.constexpr,
                        TILE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """One 64-row tile times its bank's matrix, for one block of output columns.

    ``TRANSPOSED`` reads the weight memory as ``[banks, N, K]`` and multiplies by
    its transpose, which serves the input gradient ``dy @ W^T`` with coalesced
    loads of the same weight tensor.
    """
    tile = tl.program_id(0)
    n_block = tl.program_id(1)
    bank = tl.load(bank_ptr + tile).to(tl.int64)
    rows = tile * TILE + tl.arange(0, TILE)
    cols = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((TILE, BLOCK_N), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(x_ptr + rows[:, None] * stride_xr + ks[None, :])
        if TRANSPOSED:
            b = tl.trans(tl.load(w_ptr + bank * stride_wg + cols[:, None] * stride_w1 + ks[None, :] * stride_w2))
        else:
            b = tl.load(w_ptr + bank * stride_wg + ks[:, None] * stride_w1 + cols[None, :] * stride_w2)
        acc = tl.dot(a, b, acc)
    tl.store(y_ptr + rows[:, None] * stride_yr + cols[None, :], acc.to(y_ptr.dtype.element_ty))


@triton.jit
def _tile_weight_grad_kernel(x_ptr, dy_ptr, bounds_ptr, dw_ptr,
                             stride_xr, stride_yr, stride_ws, stride_wg, stride_wk,
                             K: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr,
                             BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr):
    """One split of one bank's ``sum_tiles x[tile]^T @ dy[tile]`` for one (K, N) block."""
    bank = tl.program_id(0)
    split = tl.program_id(1)
    block = tl.program_id(2)
    k_block = block // (N // BLOCK_N)
    n_block = block % (N // BLOCK_N)
    lo = tl.load(bounds_ptr + bank)
    hi = tl.load(bounds_ptr + bank + 1)
    span = (hi - lo + SPLITS - 1) // SPLITS
    start = lo + split * span
    stop = tl.minimum(start + span, hi)
    ks = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    ns = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for tile in range(start, stop):
        rows = tile * TILE + tl.arange(0, TILE)
        a = tl.load(x_ptr + rows[:, None] * stride_xr + ks[None, :])
        g = tl.load(dy_ptr + rows[:, None] * stride_yr + ns[None, :])
        acc = tl.dot(tl.trans(a), g, acc)
    tl.store(dw_ptr + split * stride_ws + bank * stride_wg + ks[:, None] * stride_wk + ns[None, :], acc)


def _require_operands(x: Tensor, weight: Tensor, tile_bank: Tensor, bank_tiles: Tensor):
    _require(x.is_cuda and x.ndim == 2 and x.is_contiguous(), "x must be a contiguous CUDA [rows, K] matrix")
    _require(weight.is_cuda and weight.ndim == 3 and weight.is_contiguous() and weight.dtype == x.dtype,
             "weight must be a contiguous [banks, K, N] tensor of x's dtype")
    rows, k = x.shape
    banks, weight_k, n = weight.shape
    _require(weight_k == k and k % GRANULE == 0 and n % GRANULE == 0, f"K and N must be multiples of {GRANULE}")
    _require(rows % TILE == 0 and rows > 0, f"rows must be a positive multiple of the {TILE}-row tile")
    _require(tile_bank.is_cuda and tile_bank.dtype == torch.int32 and tile_bank.shape == (rows // TILE,)
             and tile_bank.is_contiguous(), "tile_bank must be int32 [rows / 64] on the device")
    _require(bank_tiles.is_cuda and bank_tiles.dtype == torch.int32 and bank_tiles.shape == (banks + 1,)
             and bank_tiles.is_contiguous(), "bank_tiles must be int32 [banks + 1] tile bounds on the device")


def _block(*dims: int) -> int:
    return 128 if all(dim % 128 == 0 for dim in dims) else GRANULE


def _tile_matmul(x: Tensor, weight: Tensor, tile_bank: Tensor, transposed: bool) -> Tensor:
    rows = x.shape[0]
    n = weight.shape[1] if transposed else weight.shape[2]
    block_n = _block(n)
    y = torch.empty(rows, n, device=x.device, dtype=x.dtype)
    _tile_matmul_kernel[(rows // TILE, n // block_n)](
        x, weight, tile_bank, y, x.stride(0), weight.stride(0), weight.stride(1), weight.stride(2), y.stride(0),
        K=x.shape[1], N=n, TRANSPOSED=transposed, TILE=TILE, BLOCK_N=block_n, BLOCK_K=BLOCK_K)
    return y


def _bank_linear_fwd_real(x: Tensor, weight: Tensor, tile_bank: Tensor, bank_tiles: Tensor) -> Tensor:
    _require_operands(x, weight, tile_bank, bank_tiles)
    return _tile_matmul(x, weight, tile_bank, transposed=False)


def _bank_linear_fwd_fake(x: Tensor, weight: Tensor, tile_bank: Tensor, bank_tiles: Tensor) -> Tensor:
    return torch.empty(x.shape[0], weight.shape[2], device=x.device, dtype=x.dtype)


bank_linear_fwd = torch.library.custom_op(f"{LIBRARY}::bank_linear_fwd", _bank_linear_fwd_real,
                                          mutates_args=(), tags=TAGS)
bank_linear_fwd.register_fake(_bank_linear_fwd_fake)


def _bank_linear_bwd_real(x: Tensor, dy: Tensor, weight: Tensor, tile_bank: Tensor, bank_tiles: Tensor) -> Tensor2:
    _require_operands(x, weight, tile_bank, bank_tiles)
    _require(dy.shape == (x.shape[0], weight.shape[2]) and dy.is_contiguous() and dy.dtype == x.dtype,
             "dy must be a contiguous [rows, N] matrix of x's dtype")
    dx = _tile_matmul(dy, weight, tile_bank, transposed=True)
    banks, k, n = weight.shape
    block = _block(k, n)
    partial = torch.empty(WEIGHT_SPLITS, banks, k, n, device=x.device, dtype=torch.float32)
    _tile_weight_grad_kernel[(banks, WEIGHT_SPLITS, (k // block) * (n // block))](
        x, dy, bank_tiles, partial, x.stride(0), dy.stride(0), partial.stride(0), partial.stride(1),
        partial.stride(2), K=k, N=n, TILE=TILE, BLOCK_K=block, BLOCK_N=block, SPLITS=WEIGHT_SPLITS)
    return dx, partial.sum(0)


def _bank_linear_bwd_fake(x: Tensor, dy: Tensor, weight: Tensor, tile_bank: Tensor, bank_tiles: Tensor) -> Tensor2:
    return torch.empty_like(x), torch.empty(weight.shape, device=weight.device, dtype=torch.float32)


bank_linear_bwd = torch.library.custom_op(f"{LIBRARY}::bank_linear_bwd", _bank_linear_bwd_real,
                                          mutates_args=(), tags=TAGS)
bank_linear_bwd.register_fake(_bank_linear_bwd_fake)


def _bank_linear_setup(ctx, inputs, output):
    x, weight, tile_bank, bank_tiles = inputs
    ctx.save_for_backward(x, weight, tile_bank, bank_tiles)


def _bank_linear_backward(ctx, dy):
    x, weight, tile_bank, bank_tiles = ctx.saved_tensors
    dx, dw = bank_linear_bwd(x, dy.contiguous(), weight, tile_bank, bank_tiles)
    return dx, dw.to(weight.dtype), None, None


bank_linear_fwd.register_autograd(_bank_linear_backward, setup_context=_bank_linear_setup)


def bank_linear(x: Tensor, weight: Tensor, tile_bank: Tensor, bank_tiles: Tensor) -> Tensor:
    """``y[r] = x[r] @ weight[tile_bank[r // 64]]`` over a packed pool.

    ``tile_bank`` names every 64-row tile's bank; ``bank_tiles`` holds each
    bank's contiguous tile range ``[bank_tiles[j], bank_tiles[j + 1])`` and is
    used by the weight gradient. Tiles outside every range (the filler) must be
    zero rows: they take some bank's matrix in the forward and contribute no
    weight gradient. The weight is used in ``x``'s dtype; its gradient is
    accumulated in fp32 and returned in the weight's dtype.
    """
    return bank_linear_fwd(x, weight, tile_bank, bank_tiles)

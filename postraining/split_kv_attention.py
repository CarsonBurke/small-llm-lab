# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
"""SM120 MiniCPM decode: packed FA4 split-4 with FP32 partial outputs.

Retains the installed FA4 main loop, including BF16 probability operands and
FP32 accumulators. Only the output epilogue changes: normalized partials stay
FP32 until FA4's log-sum-exp merge writes the final BF16 output. Single-query
attention needs no causal mask: device lengths already exclude future KV.
The 16x32, single-warp tile limits padding for eight packed GQA query rows.

Specializations must be warmed before CUDA graph capture. No backward contract.
"""

from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
from flash_attn.cute.interface import _flash_attn_fwd_combine
from flash_attn.cute.pack_gqa import PackGQA
from flash_attn.cute.utils import AuxData
from quack import layout_utils
import torch


_SPLITS = 4
_TILE_N = 32


class _PartialFA4(FlashAttentionForwardSm120):
    def __init__(self):
        super().__init__(
            cutlass.BFloat16,
            128,
            128,
            8,
            is_causal=False,
            is_local=False,
            pack_gqa=True,
            tile_m=16,
            tile_n=_TILE_N,
            num_stages=1,
            num_threads=32,
            Q_in_regs=False,
        )

    def _check_type(self, mQ_type, mK_type, mV_type, mO_type, *other_types):
        if mO_type != Float32:
            raise TypeError("FP32 partial output required")
        super()._check_type(mQ_type, mK_type, mV_type, mQ_type, *other_types)

    @cute.jit
    def epilogue(
        self,
        acc_O,
        lse,
        mO,
        mLSE,
        sO,
        seqlen,
        gmem_tiled_copy_O,
        tma_atom_O,
        tiled_mma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        # Every accumulator has one owner. Store it directly, bypassing only
        # FA4's BF16 shared-memory output staging, not its attention arithmetic.
        thr_mma = tiled_mma.get_slice(tidx)
        coordinates = thr_mma.partition_C(
            cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        )
        coords_mn = layout_utils.reshape_acc_to_mn(coordinates)
        acc_mn = layout_utils.reshape_acc_to_mn(acc_O)
        for row in cutlass.range_constexpr(cute.size(acc_mn.shape[0])):
            for col in cutlass.range_constexpr(cute.size(acc_mn.shape[1])):
                qrow = m_block * self.tile_m + coords_mn[row, col][0]
                d = coords_mn[row, col][1]
                if qrow < seqlen.seqlen_q * self.qhead_per_kvhead and d < 128:
                    mO[((qrow % 8, qrow // 8), d, head_idx, batch_idx)] = acc_mn[
                        row, col
                    ]
        if const_expr(mLSE is not None):
            pack = PackGQA(self.tile_m, self.tile_hdimv, False, 8)
            pack.store_LSE(
                mLSE[None, head_idx, batch_idx],
                lse,
                tiled_mma,
                tidx,
                m_block,
                seqlen.seqlen_q,
            )


@lru_cache(maxsize=None)
def _metadata(batch, capacity, device):
    return (
        torch.arange(_SPLITS, device=device, dtype=torch.int32)[None, :],
        torch.arange(batch, device=device, dtype=torch.int32)[:, None] * capacity,
    )


@torch.compile(fullgraph=True)
def _prepare(query, lengths, split_ids, batch_offsets, capacity):
    batch = query.shape[0]
    # Align starts to K tiles; empty partitions remain truly empty.
    span = ((lengths + _TILE_N * _SPLITS - 1) // (_TILE_N * _SPLITS)) * _TILE_N
    starts = torch.minimum(span[:, None] * split_ids, lengths[:, None])
    live = torch.minimum(span[:, None], lengths[:, None] - starts).reshape(-1)
    offsets = torch.cat(
        (
            (batch_offsets + starts).reshape(-1),
            torch.full((1,), batch * capacity, device=query.device, dtype=torch.int32),
        )
    )
    queries = (
        query[:, None]
        .expand(batch, _SPLITS, 1, 16, 128)
        .reshape(batch * _SPLITS, 1, 16, 128)
    )
    return queries, offsets, live


_kernels = {}


def split_kv_attention(query, key, value, lengths, scale):
    """Decode BF16 [B,1,16,128] against contiguous [B,C,2,128] KV.

    Caller supplies CUDA int32 lengths in [1,C]; retired lanes use length one.
    Offsets and live lengths are recomputed on-device on every graph replay,
    including retirement and refill. KV is viewed, never replicated or copied.
    """
    batch, capacity = key.shape[:2]
    split_ids, batch_offsets = _metadata(batch, capacity, query.device)
    queries, offsets, live = _prepare(
        query, lengths, split_ids, batch_offsets, capacity
    )
    keys = key.view(batch * capacity, 2, 128)
    values = value.view(batch * capacity, 2, 128)
    partial = torch.empty(
        (batch * _SPLITS, 1, 16, 128), dtype=torch.float32, device=query.device
    )
    lse = torch.empty(
        (batch * _SPLITS, 16, 1), dtype=torch.float32, device=query.device
    )
    cache_key = (query.device, batch, capacity, float(scale))
    args = (
        queries,
        keys,
        values,
        partial,
        lse,
        float(scale),
        None,
        offsets,
        None,
        live,
        None,
        None,
        None,
        None,
        None,
        AuxData(),
        None,
        None,
    )
    if cache_key not in _kernels:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm FA4 split specialization before capture")
        compile_args = [to_cute_tensor(t) for t in (queries, keys, values, partial)]
        compile_args.extend(
            (
                to_cute_tensor(lse, assumed_align=4),
                float(scale),
                None,
                to_cute_tensor(offsets, assumed_align=4, leading_dim=0),
                None,
                to_cute_tensor(live, assumed_align=4, leading_dim=0),
                None,
                None,
                None,
                None,
                None,
                AuxData(),
                None,
                None,
            )
        )
        _kernels[cache_key] = cute.compile(
            _PartialFA4(),
            *compile_args,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _kernels[cache_key](*args)
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    _flash_attn_fwd_combine(
        partial.view(batch, _SPLITS, 1, 16, 128).permute(1, 0, 2, 3, 4),
        lse.view(batch, _SPLITS, 16, 1).permute(1, 0, 3, 2),
        output,
        _arch=120,
    )
    return output

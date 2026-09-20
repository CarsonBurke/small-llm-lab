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

from postraining.split_kv_plan import (
    SPLITS as _SPLITS,
    TILE_N as _TILE_N,
    plan_split_kv,
    split_kv_metadata,
    split_kv_plan_shapes,
)


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
    return split_kv_metadata(batch, capacity, device)


_kernels = {}


def split_kv_attention(query, key, value, lengths, scale, offsets=None, live=None):
    """Decode BF16 [B,1,16,128] against contiguous [B,C,2,128] KV.

    Caller supplies CUDA int32 lengths in [1,C]; retired lanes use length one.
    The partition plan (``offsets``, ``live``) depends only on ``lengths``:
    a caller that shares one plan across layers passes it in, otherwise it is
    derived here. Either way it is recomputed on-device on every graph
    replay, including retirement and refill. KV is viewed, never copied.
    """
    batch, capacity = key.shape[:2]
    if (offsets is None) != (live is None):
        raise ValueError("split-KV plan needs both offsets and live lengths")
    if offsets is None:
        offsets, live = plan_split_kv(lengths, *_metadata(batch, capacity, query.device))
    expected_shapes = split_kv_plan_shapes(batch)
    if (
        offsets.shape != (expected_shapes[0],)
        or live.shape != (expected_shapes[1],)
        or offsets.dtype != torch.int32
        or live.dtype != torch.int32
    ):
        raise ValueError("split-KV plan shape or dtype differs from the KV batch")
    queries = (
        query[:, None]
        .expand(batch, _SPLITS, 1, 16, 128)
        .reshape(batch * _SPLITS, 1, 16, 128)
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

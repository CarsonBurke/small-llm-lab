"""Adapt upstream GDN-2's helper keywords to installed fla-core 0.5.2.

The installed common-delta and GLA kernels already consume base-two log gates
and evaluate exp2. Therefore only use_exp2=True is supported. State layout is
the same choice under a different name; no gates or tensors are transformed.
See ../provenance.json for the upstream pin and this integration patch.
"""

from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu as _bwd_dhu,
    chunk_gated_delta_rule_fwd_h as _fwd_h,
)
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk as _fwd_o_gk


def _keywords(use_exp2, transpose_state_layout, kwargs):
    if use_exp2 is not True:
        raise ValueError("fla-core 0.5.2 helpers require base-two log gates (use_exp2=True)")
    if "state_v_first" in kwargs:
        raise ValueError("Provide only the upstream transpose_state_layout keyword")
    return {**kwargs, "state_v_first": transpose_state_layout}


def chunk_gated_delta_rule_fwd_h(*args, use_exp2=True, transpose_state_layout=False, **kwargs):
    return _fwd_h(*args, **_keywords(use_exp2, transpose_state_layout, kwargs))


def chunk_gated_delta_rule_bwd_dhu(*args, use_exp2=True, transpose_state_layout=False, **kwargs):
    return _bwd_dhu(*args, **_keywords(use_exp2, transpose_state_layout, kwargs))


def chunk_gla_fwd_o_gk(*args, use_exp2=True, transpose_state_layout=False, **kwargs):
    return _fwd_o_gk(*args, **_keywords(use_exp2, transpose_state_layout, kwargs))

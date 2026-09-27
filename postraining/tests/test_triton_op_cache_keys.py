"""Every repository ``triton_op`` exposes its kernels to the compile caches.

AOTAutograd's cache keys a graph that calls a ``triton_op`` on the source of
the kernels torch finds inside the op, by matching ``wrap_triton(<name>)``
spellings. ``torch.library.wrap_triton(...)`` or a kernel behind a ``cast``
is invisible to that match, so editing the kernel leaves the key unchanged
and a warm cache serves the previous kernel's compiled graph.
"""

import importlib

import pytest

triton_library = pytest.importorskip("torch._library.triton")

MODULE_OPS = {
    "postraining.decode_attention": {
        "nanogpt_attn::ranged_decode": {
            "_ranged_decode_kernel",
            "_merge_splits_kernel",
        },
    },
    "postraining.nucleus_threshold": {
        "nanogpt_sampling::nucleus_threshold": {
            "_stats_partial_kernel",
            "_stats_combine_kernel",
            "_cut_partial_kernel",
            "_cut_combine_kernel",
        },
    },
    "postraining.fast_inference": {
        "parameter_golf::w8a16_linear": {"_w8a16_linear_kernel"},
    },
    "pretraining.nanogpt_mini.kda_decode_kernel": {
        "nanogpt_kda::decode_step": {"_kda_decode_step_kernel"},
    },
}


@pytest.mark.parametrize("module", sorted(MODULE_OPS))
def test_triton_op_kernels_are_in_the_cache_key(module):
    importlib.import_module(module)
    for op, kernels in MODULE_OPS[module].items():
        found = {
            kernel.fn.__name__ if hasattr(kernel, "fn") else kernel.__name__
            for kernel in triton_library.get_triton_kernels_for_op(op)
        }
        assert found == kernels, op

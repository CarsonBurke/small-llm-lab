"""Every repository ``triton_op`` is part of the compile cache key.

AOTAutograd's cache keys a graph that calls a ``triton_op`` on the source of
the kernels torch finds inside the op, by matching ``wrap_triton(<name>)``
spellings. ``torch.library.wrap_triton(...)`` or a kernel behind a ``cast``
is invisible to that match, so editing the kernel leaves the key unchanged
and a warm cache serves the previous kernel's compiled graph. The op's
Python body is never in that key, so each module also folds its own source
into ``torch.compiler.config.cache_key_tag``.
"""

import hashlib
import importlib
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]

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


def test_every_repository_triton_op_module_is_covered():
    """A new op module must join MODULE_OPS and key the caches on itself."""
    defining = {
        ".".join(path.relative_to(REPO).with_suffix("").parts)
        for root in ("postraining", "pretraining")
        for path in (REPO / root).rglob("*.py")
        if "tests" not in path.parts
        and "triton_op(" in path.read_text(encoding="utf-8")
    }
    assert defining == set(MODULE_OPS)


@pytest.mark.parametrize("module", sorted(MODULE_OPS))
def test_triton_op_kernels_are_in_the_cache_key(module):
    importlib.import_module(module)
    for op, kernels in MODULE_OPS[module].items():
        found = {
            kernel.fn.__name__ if hasattr(kernel, "fn") else kernel.__name__
            for kernel in triton_library.get_triton_kernels_for_op(op)
        }
        assert found == kernels, op


@pytest.mark.parametrize("module", sorted(MODULE_OPS))
def test_triton_op_module_source_is_in_the_cache_key_tag(module):
    imported = importlib.import_module(module)
    digest = hashlib.sha256(Path(imported.__file__).read_bytes()).hexdigest()[:16]
    assert f"{module}={digest}" in torch.compiler.config.cache_key_tag.split(";")


def test_cache_key_tag_changes_with_the_source(tmp_path):
    from pretraining.nanogpt_mini import triton_op_cache

    before = torch.compiler.config.cache_key_tag
    source = tmp_path / "op.py"
    try:
        source.write_text("GRID = 1\n")
        triton_op_cache.key_compile_caches_on_source("probe", str(source))
        first = torch.compiler.config.cache_key_tag
        source.write_text("GRID = 2\n")
        triton_op_cache.key_compile_caches_on_source("probe", str(source))
        second = torch.compiler.config.cache_key_tag
        assert second != first
        # Every other module's digest (and any environment tag) is kept.
        assert set(before.split(";")) - {""} <= set(second.split(";"))
    finally:
        triton_op_cache._DIGESTS.pop("probe", None)
        torch.compiler.config.cache_key_tag = before

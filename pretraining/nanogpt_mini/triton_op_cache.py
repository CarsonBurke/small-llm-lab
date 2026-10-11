"""Key torch.compile's caches on the Python source of this repository's triton_ops.

AOTAutograd's cache keys a graph that calls a ``triton_op`` on the source of
the op's Triton kernels (``torch._library.triton.get_triton_kernels_for_op``),
not on the op's Python body, which chooses grids, split counts and scratch
buffers. An edit to that body leaves the key unchanged, and a warm cache
keeps serving the graph traced from the old body: on 2026-09-26 a
dynamic-shape compile of ``nucleus_threshold`` ran a superseded split rule
while eager calls ran the new one.

Each module defining a ``triton_op`` therefore folds a digest of its own
bytes into ``torch.compiler.config.cache_key_tag``, which every compile cache
key includes. Editing such a module invalidates compiled graphs once; nothing
else does.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

# Whatever tag the environment set (TORCH_COMPILE_CACHE_KEY_TAG) stays first.
_BASE_TAG = torch.compiler.config.cache_key_tag
_DIGESTS: dict[str, str] = {}


def key_compile_caches_on_source(module_name: str, module_file: str) -> None:
    """Fold ``module_file``'s bytes into the compile cache key tag."""
    digest = hashlib.sha256(Path(module_file).read_bytes()).hexdigest()[:16]
    _DIGESTS[module_name] = digest
    parts = [_BASE_TAG] if _BASE_TAG else []
    parts.extend(f"{name}={_DIGESTS[name]}" for name in sorted(_DIGESTS))
    torch.compiler.config.cache_key_tag = ";".join(parts)

"""VAPO post-training for the fresh LeJEPA language model."""

import os

# Variable-shape rollout batches fragment the caching allocator: job 143 died
# with 12 GiB reserved-but-unallocated.  Expandable segments let reserved
# blocks grow instead of stranding, and the allocator reads this at first
# CUDA allocation, so the package import is early enough for every
# `python3 -m postraining.*` entry point.  setdefault keeps an explicit
# environment override in charge.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


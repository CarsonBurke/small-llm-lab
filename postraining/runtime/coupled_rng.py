"""Stateless, counter-addressed CUDA randomness for paired controller rollouts.

A real draw's address is (seed64, component32, position32, pair32, role32).
The four counter fields are separate Philox words, never a flattened offset.
Use role 1 for Gaussian actions, 2 for gate uniforms, and 3 for answer uniforms.
Different roles have disjoint counters, not a mathematical independence proof.
Normal draws are unsigned: the caller applies +1/-1 to even/odd logical lanes.

Real pair IDs and positions must be in [0, 2**32). Negative indices denote
padding and produce zero. Positive overflow produces NaN rather than silently
wrapping into another real counter. Value checks never synchronize to the host.
A signed int64 seed supplies all 64 key bits, including negative seed values.

This uses installed Triton's Philox4x32-10 and exactly its float32 uniform and
Box-Muller conversion convention, NOT ideal continuous randomness. Uniforms
are in [0, 1), including zero; signed-folded uint32 conversion has a 31-bit
integer source and FP32 rounding. Box-Muller floors its radial uniform at
1e-7, as tl.randn does, bounding radius to approximately 5.678. The Gaussian
output is therefore a finite-precision approximation to N(0,1), with resolved
tails limited by that convention. No sigma scaling or sign is applied here.

Imports register kernels only. Warm up each shape/stride specialization before
CUDA capture; outputs use PyTorch's capture-safe allocator, with no persistent
cache or RNG state. Seed and index buffer CONTENTS may change between replays.
Qualification CLI: scripts/minicpm_coupled_rng.py (queue through mlq).
"""

import hashlib
import math
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language import random as tl_random


_UINT32_LIMIT = 1 << 32


@triton.jit
def _coupled_kernel(
    Seed, Pairs, Positions, Output,
    N: tl.constexpr, D: tl.constexpr, PAIR_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr, ROLE: tl.constexpr,
    NORMAL: tl.constexpr, BLOCK: tl.constexpr,
):
    # The launch offset addresses output memory only, not the random counter.
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offset // D
    component = (offset % D).to(tl.uint32)
    mask = row < N
    pair = tl.load(Pairs + row * PAIR_STRIDE, mask=mask, other=-1)
    position = tl.load(Positions + row * POSITION_STRIDE, mask=mask, other=-1)
    seed = tl.load(Seed).to(tl.uint64)
    # Explicit words preserve addresses when rows are reordered or padded.
    i1, i2, _, _ = tl_random.philox(
        seed, component, position.to(tl.uint32), pair.to(tl.uint32),
        tl.full((BLOCK,), ROLE, tl.uint32), n_rounds=10,
    )
    u1 = tl_random.uint_to_uniform_float(i1)
    if NORMAL:
        u2 = tl_random.uint_to_uniform_float(i2)
        value, _ = tl_random.pair_uniform_to_normal(u1, u2)
    else:
        value = u1
    padding = (pair < 0) | (position < 0)
    overflow = (pair >= 4294967296) | (position >= 4294967296)
    value = tl.where(padding, 0.0, tl.where(overflow, float("nan"), value))
    tl.store(Output + offset, value, mask=mask)


def _validate_metadata(seed, pair_ids, positions, dimension, role):
    """Metadata only: safe under fullgraph tracing and graph capture."""
    if seed.device.type != "cuda":
        raise ValueError("coupled RNG requires CUDA; there is no CPU fallback")
    if seed.dtype != torch.int64 or seed.ndim != 0:
        raise ValueError("seed must be a scalar CUDA int64 tensor")
    for name, indices in (("pair_ids", pair_ids), ("positions", positions)):
        if indices.device != seed.device or indices.dtype != torch.int64 or indices.ndim != 1:
            raise ValueError(f"{name} must be a same-device CUDA int64 vector")
    if pair_ids.shape != positions.shape:
        raise ValueError("pair_ids and positions must have identical shapes")
    if isinstance(dimension, bool) or not isinstance(dimension, (int, torch.SymInt)) or not 1 <= dimension <= _UINT32_LIMIT:
        raise ValueError("dimension must be an integer in [1, 2**32]")
    if type(role) is not int or not 0 <= role < _UINT32_LIMIT:
        raise ValueError("role must be an integer in [0, 2**32)")


@torch.library.custom_op("minicpm_coupled_rng::draw", mutates_args=(), device_types="cuda")
def _draw(
    seed: torch.Tensor, pair_ids: torch.Tensor, positions: torch.Tensor,
    dimension: int, role: int, normal: bool,
) -> torch.Tensor:
    _validate_metadata(seed, pair_ids, positions, dimension, role)
    output = torch.empty((pair_ids.numel(), dimension), device=seed.device, dtype=torch.float32)
    if pair_ids.numel():
        with torch.cuda.device(seed.device):
            _coupled_kernel[(triton.cdiv(output.numel(), 256),)](
                seed, pair_ids, positions, output, pair_ids.numel(), dimension,
                pair_ids.stride(0), positions.stride(0), role, normal, 256,
            )
    return output


@_draw.register_fake
def _draw_fake(seed, pair_ids, positions, dimension, role, normal):
    _validate_metadata(seed, pair_ids, positions, dimension, role)
    return torch.empty((pair_ids.numel(), dimension), device=seed.device, dtype=torch.float32)


def coupled_normal(
    seed: torch.Tensor, pair_ids: torch.Tensor, positions: torch.Tensor,
    *, dimension: int, role: int,
) -> torch.Tensor:
    """Return unsigned FP32 [N,D] draws, addressed independently of row order."""
    _validate_metadata(seed, pair_ids, positions, dimension, role)
    return _draw(seed, pair_ids, positions, dimension, role, True)


def coupled_uniform(
    seed: torch.Tensor, pair_ids: torch.Tensor, positions: torch.Tensor, *, role: int,
) -> torch.Tensor:
    """Return FP32 [N] uniforms in [0,1), using component word zero."""
    _validate_metadata(seed, pair_ids, positions, 1, role)
    return _draw(seed, pair_ids, positions, 1, role, False).squeeze(1)


def rng_metadata():
    """Host-side provenance for reports; do not call inside compiled/captured code."""
    source = Path(__file__).resolve()
    triton_source = Path(tl_random.__file__).resolve()
    return {
        "algorithm": "Philox4x32-10",
        "counter_words": ["component_uint32", "response_position_uint32", "pair_id_uint32", "role_uint32"],
        "key": "scalar CUDA int64 bit pattern, low32/high32 Philox key words",
        "real_index_range": [0, _UINT32_LIMIT - 1],
        "padding": "negative pair or position -> zero",
        "overflow": "non-padding pair or position >= 2**32 -> NaN",
        "roles": {"gaussian": 1, "gate": 2, "answer": 3},
        "role_independence": "disjoint counter domains; pseudorandom independence assumption, not a proof",
        "uniform": "Triton uint_to_uniform_float: signed-folded 31-bit source, FP32 rounded, [0,1)",
        "normal": "Triton pair_uniform_to_normal first output; radial uniform max(u,1e-7)",
        "normal_radius_bound_approx": math.sqrt(-2 * math.log(1e-7)),
        "marginal_caveat": "finite-precision normal approximation matching tl.randn conversion, not ideal real-valued exactness",
        "antithetic_sign": "not applied; caller uses +1 even logical lane, -1 odd logical lane",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "triton_random_source": str(triton_source),
        "triton_random_source_sha256": hashlib.sha256(triton_source.read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "triton_version": str(triton.__version__),
    }

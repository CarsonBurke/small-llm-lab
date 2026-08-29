"""pretraining/nanogpt_mini/nanogpt_mini_dblock.py

DiffusionBlocks (Shing et al., ICLR 2026, arXiv:2506.14202) machinery for the
KDA-mixer nanogpt-mini trainer, importable and CPU-testable without FLA.

The framework partitions a residual network into blocks that are trained
independently as denoisers over the target-embedding space. For the
autoregressive case the target of position ``i`` is ``targets[i]`` (the next
token): its unit-L2-normalized embedding ``y_i`` is noised to
``z_sigma = y + sigma * eps`` and each block learns to recover ``y_i`` from
``z_sigma,i`` conditioned on the *clean* token prefix ``inputs[0..i]``. Blocks
own equal probability-mass slices of the EDM lognormal noise distribution;
the earliest layers denoise the highest band because the residual stream
integrates the reverse diffusion from noise to data.

Layout contract used by the trainer (one forward, causal-consistent):

  - The residual stream is the concatenation ``[clean(T), noisy(T)]``.
  - Dense attention: clean queries are ordinarily causal over clean keys;
    the noisy query for position ``i`` sees clean keys ``j <= i`` plus itself
    (it acts at absolute position ``i + 1``), and never other noisy tokens.
  - KDA: q/k/v/gate/beta are interleaved ``[c_0, n_0, c_1, n_1, ...]`` for the
    recurrence, with noisy slots made read-only by masking their decay and
    write logits (decay -> 1, beta -> 0). The clean state trajectory is then
    exactly the clean-only trajectory, and slot ``n_i`` reads the state after
    ``c_0..c_i``.
  - Short convolutions: the noisy branch window for position ``i`` is
    ``[c_{i-2}, c_{i-1}, c_i, n_i]`` — the window the original network would
    see if the next input token were the noisy embedding.

Read-only KDA slots deliberately drop the delta-rule self-read that a
committed step would add (the token's own content still enters through the
residual stream and its conv tap); dense attention keeps the self key. Both
choices are identical at training and inference time, which is the invariant
that matters.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import torch
from torch import Tensor, nn

_STANDARD_NORMAL = NormalDist()

# Logit magnitude that drives sigmoid to numerical zero in fp32 while staying
# representable in bf16. Used to force noisy-slot KDA decay to exp(0)=1 and
# write strength to 0 inside FLA's in-kernel sigmoid parameterizations.
READ_ONLY_LOGIT = -30000.0


########################################
#         Noise-level partition        #
########################################


def lognormal_cdf(sigma: float, p_mean: float, p_std: float) -> float:
    return _STANDARD_NORMAL.cdf((math.log(sigma) - p_mean) / p_std)


def lognormal_icdf(mass: float, p_mean: float, p_std: float) -> float:
    return math.exp(p_mean + p_std * _STANDARD_NORMAL.inv_cdf(mass))


def equal_mass_sigma_boundaries(
    num_blocks: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> list[float]:
    """Ascending ``num_blocks + 1`` boundaries splitting ``[sigma_min,
    sigma_max]`` into equal slices of the lognormal training mass."""
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    if not 0 < sigma_min < sigma_max:
        raise ValueError(
            f"need 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}"
        )
    cdf_min = lognormal_cdf(sigma_min, p_mean, p_std)
    cdf_max = lognormal_cdf(sigma_max, p_mean, p_std)
    boundaries = [sigma_min]
    for index in range(1, num_blocks):
        mass = cdf_min + (cdf_max - cdf_min) * index / num_blocks
        boundaries.append(lognormal_icdf(mass, p_mean, p_std))
    boundaries.append(sigma_max)
    return boundaries


def block_sigma_range(
    boundaries: list[float],
    block_index: int,
    gamma: float,
) -> tuple[float, float]:
    """Training sigma range of layer-order block ``block_index``.

    Block 0 (the earliest layers) owns the highest-noise slice; ``gamma``
    log-extends both ends (the paper's overlap) clipped to the global range.
    """
    num_blocks = len(boundaries) - 1
    if not 0 <= block_index < num_blocks:
        raise ValueError(
            f"block_index {block_index} outside [0, {num_blocks})"
        )
    if gamma < 0:
        raise ValueError(f"gamma must be nonnegative, got {gamma}")
    low = boundaries[num_blocks - 1 - block_index]
    high = boundaries[num_blocks - block_index]
    if gamma:
        log_range = math.log(high) - math.log(low)
        low = max(math.exp(math.log(low) - gamma * log_range), boundaries[0])
        high = min(math.exp(math.log(high) + gamma * log_range), boundaries[-1])
    return low, high


def sigma_to_block(sigma: float, boundaries: list[float]) -> int:
    """Layer-order block index owning noise level ``sigma`` (no overlap)."""
    num_blocks = len(boundaries) - 1
    for index in range(1, num_blocks):
        if sigma < boundaries[index]:
            return num_blocks - index
    return 0


def sample_block_sigmas(
    low: float,
    high: float,
    num_samples: int,
    p_mean: float,
    p_std: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample sigmas from the lognormal restricted to ``[low, high]``.

    Inverse-CDF sampling in fp64 on the sampling device so the CUDA RNG
    stream (checkpointed on resume) governs the draw.
    """
    cdf_low = lognormal_cdf(low, p_mean, p_std)
    cdf_high = lognormal_cdf(high, p_mean, p_std)
    uniform = torch.rand(
        num_samples, dtype=torch.float64, device=device, generator=generator
    )
    mass = cdf_low + (cdf_high - cdf_low) * uniform
    # Rational-approximation probit (Acklam) is exact enough here (relative
    # error ~1e-9) and keeps the draw on-device and generator-driven.
    normal_quantile = _probit(mass)
    return (p_mean + p_std * normal_quantile).exp().float()


def _probit(p: Tensor) -> Tensor:
    """Acklam's inverse normal CDF approximation, elementwise fp64."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p = p.clamp(1e-12, 1 - 1e-12)
    p_low, p_high = 0.02425, 1 - 0.02425

    q_low = (-2 * p.clamp(max=p_low).log()).sqrt()
    x_low = (
        (((((c[0] * q_low + c[1]) * q_low + c[2]) * q_low + c[3]) * q_low + c[4]) * q_low + c[5])
        / ((((d[0] * q_low + d[1]) * q_low + d[2]) * q_low + d[3]) * q_low + 1)
    )
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    x_mid = (
        (((((a[0] * r_mid + a[1]) * r_mid + a[2]) * r_mid + a[3]) * r_mid + a[4]) * r_mid + a[5]) * q_mid
        / (((((b[0] * r_mid + b[1]) * r_mid + b[2]) * r_mid + b[3]) * r_mid + b[4]) * r_mid + 1)
    )
    q_high = (-2 * (1 - p.clamp(min=p_high)).log()).sqrt()
    x_high = -(
        (((((c[0] * q_high + c[1]) * q_high + c[2]) * q_high + c[3]) * q_high + c[4]) * q_high + c[5])
        / ((((d[0] * q_high + d[1]) * q_high + d[2]) * q_high + d[3]) * q_high + 1)
    )
    return torch.where(p < p_low, x_low, torch.where(p > p_high, x_high, x_mid))


def inference_sigma_schedule(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> list[float]:
    """Descending equi-mass sigma grid; ``num_steps == num_blocks`` routes
    exactly one denoise call to every block, highest band first."""
    if num_steps < 2:
        raise ValueError(f"num_steps must be at least 2, got {num_steps}")
    cdf_min = lognormal_cdf(sigma_min, p_mean, p_std)
    cdf_max = lognormal_cdf(sigma_max, p_mean, p_std)
    grid = []
    for index in range(num_steps):
        mass = cdf_min + (cdf_max - cdf_min) * index / (num_steps - 1)
        grid.append(lognormal_icdf(mass, p_mean, p_std))
    grid[0], grid[-1] = sigma_min, sigma_max
    return grid[::-1]


########################################
#         EDM preconditioning          #
########################################


def edm_coefficients(sigma: Tensor, sigma_data: float) -> dict[str, Tensor]:
    """EDM (Karras et al. 2022) scalars, elementwise fp32 over ``sigma``."""
    sigma = sigma.float()
    variance = sigma.square() + sigma_data**2
    return {
        "c_skip": sigma_data**2 / variance,
        "c_out": sigma * sigma_data * variance.rsqrt(),
        "c_in": variance.rsqrt(),
        "c_noise": 0.25 * sigma.log(),
        "weight": variance / (sigma * sigma_data).square(),
    }


def normalized_embedding_table(embed_weight: Tensor) -> Tensor:
    """Unit-L2 rows (Dieleman-style anti-collapse), fp32."""
    weight = embed_weight.float()
    return weight * weight.square().sum(-1, keepdim=True).clamp(min=1e-12).rsqrt()


########################################
#          Layer partitioning          #
########################################


def partition_layers_by_dense(
    delta_layer_indices: frozenset[int] | set[int],
    num_layers: int,
    num_blocks: int | None = None,
) -> list[list[int]]:
    """Split layers into diffusion blocks: every maximal run of KDA mixers
    together with the dense layer that closes it forms one group. The
    schedule must end on a dense layer so no trailing mixers are orphaned.
    ``num_blocks`` merges adjacent groups evenly when fewer, larger blocks
    are wanted; it must divide the group count.
    """
    if not set(delta_layer_indices) <= set(range(num_layers)):
        raise ValueError("delta_layer_indices outside the layer range")
    if (num_layers - 1) in delta_layer_indices:
        raise ValueError(
            "the last layer must be dense: a trailing KDA run has no dense "
            "layer to close its diffusion block"
        )
    groups: list[list[int]] = []
    current: list[int] = []
    for layer_index in range(num_layers):
        current.append(layer_index)
        if layer_index not in delta_layer_indices:
            groups.append(current)
            current = []
    assert not current
    if num_blocks is None or num_blocks == len(groups):
        return groups
    if num_blocks <= 0 or len(groups) % num_blocks:
        raise ValueError(
            f"num_blocks={num_blocks} must evenly divide the "
            f"{len(groups)} dense-terminated groups"
        )
    merge = len(groups) // num_blocks
    return [
        [layer for group in groups[i * merge:(i + 1) * merge] for layer in group]
        for i in range(num_blocks)
    ]


########################################
#           Two-stream helpers         #
########################################


def dblock_mask_mod(seq_len: int):
    """flex_attention mask for the ``[clean(T), noisy(T)]`` layout."""

    def mask_mod(batch, head, q_idx, kv_idx):
        q_noisy = q_idx >= seq_len
        k_clean = kv_idx < seq_len
        q_pos = torch.where(q_noisy, q_idx - seq_len, q_idx)
        clean_causal = (~q_noisy) & k_clean & (kv_idx <= q_idx)
        noisy_reads_clean = q_noisy & k_clean & (kv_idx <= q_pos)
        noisy_self = q_noisy & (kv_idx == q_idx)
        return clean_causal | noisy_reads_clean | noisy_self

    return mask_mod


def dblock_positions(seq_len: int, device: torch.device) -> Tensor:
    """Rotary positions for the concatenated layout: clean token ``i`` acts
    at position ``i``; the noisy token for target ``i`` acts at ``i + 1``."""
    clean = torch.arange(seq_len, dtype=torch.float32, device=device)
    return torch.cat((clean, clean + 1))


def interleave_streams(clean: Tensor, noisy: Tensor) -> Tensor:
    """``[B, T, ...] x 2 -> [B, 2T, ...]`` as ``c_0, n_0, c_1, n_1, ...``."""
    stacked = torch.stack((clean, noisy), dim=2)
    return stacked.reshape(clean.shape[0], 2 * clean.shape[1], *clean.shape[2:])


def deinterleave_streams(mixed: Tensor) -> tuple[Tensor, Tensor]:
    return mixed[:, 0::2], mixed[:, 1::2]


def noisy_conv_preactivation(
    clean_inputs: Tensor, noisy_inputs: Tensor, weight: Tensor
) -> Tensor:
    """Depthwise causal-conv preactivation for the noisy branch.

    ``weight`` is FLA's ``[D, 1, W]`` layout with the newest tap last. The
    noisy window at position ``i`` is ``[c_{i-W+2}, ..., c_i, n_i]``: the
    clean past shifted one slot toward the present plus the noisy tap.
    """
    width = weight.size(-1)
    clean_part = torch.nn.functional.conv1d(
        clean_inputs.transpose(1, 2),
        weight[..., : width - 1].type_as(clean_inputs),
        groups=weight.size(0),
        padding=width - 2,
    )[..., : clean_inputs.size(1)].transpose(1, 2)
    return clean_part + noisy_inputs * weight[:, 0, -1].type_as(noisy_inputs)


########################################
#         Noise conditioning           #
########################################


class TimestepEmbedder(nn.Module):
    """DiT-style sinusoidal noise-level embedder feeding the AdaLN heads."""

    def __init__(self, cond_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.fc1 = nn.Linear(freq_dim, cond_dim, bias=True)
        self.fc2 = nn.Linear(cond_dim, cond_dim, bias=True)
        half = freq_dim // 2
        self.register_buffer(
            "freqs",
            torch.exp(
                -math.log(10000)
                * torch.arange(half, dtype=torch.float32)
                / half
            ),
        )

    def forward(self, c_noise: Tensor) -> Tensor:
        angles = c_noise.float()[:, None] * self.freqs[None, :]
        features = torch.cat((angles.cos(), angles.sin()), dim=-1)
        hidden = torch.nn.functional.silu(self.fc1(features))
        return self.fc2(hidden)

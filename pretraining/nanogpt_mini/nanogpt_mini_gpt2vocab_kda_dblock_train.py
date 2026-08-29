"""pretraining/nanogpt_mini/nanogpt_mini_gpt2vocab_kda_dblock_train.py

DiffusionBlocks (arXiv:2506.14202, ICLR 2026) conversion of the KDA 3:1
hybrid trainer ``nanogpt_mini_gpt2vocab_kda_3to1_pm_train.py``.

The layer stack is partitioned into diffusion blocks along the natural
architectural seams: every maximal run of KDA mixers plus the dense layer
that closes it is one block (the default 24-layer 3:1 schedule yields six
blocks of ``KDA,KDA,KDA,dense``). Each block independently learns to denoise
the unit-L2-normalized embedding of the *next* token within its assigned
equal-probability-mass slice of the EDM lognormal noise distribution,
conditioned on the clean token prefix. One block trains per optimizer step,
so gradients, activations, and optimizer traffic cover ``num_layers /
num_blocks`` layers instead of the full stack.

Causal consistency with a recurrent mixer in the stack (the paper's
sequence-concatenation trick assumes dense attention):

  - The residual stream is ``[clean(T), noisy(T)]``. The noisy slot for
    position ``i`` holds the noised embedding of ``targets[i]`` and acts at
    absolute position ``i + 1``.
  - Dense layers use flex_attention: clean queries are causal over clean
    keys; noisy query ``i`` sees clean keys ``j <= i`` plus itself and never
    another noisy token.
  - KDA layers interleave the streams ``[c_0, n_0, c_1, n_1, ...]`` through
    the released chunk kernel with noisy slots made read-only (decay logits
    and write-strength logits forced to numerical -inf, so decay = 1 and
    beta = 0). The clean state trajectory is bit-identical to a clean-only
    pass and slot ``n_i`` reads exactly the state after ``c_0..c_i``.
  - KDA short convolutions give the noisy branch the window
    ``[c_{i-2}, c_{i-1}, c_i, n_i]``.
  - AdaLN shift/scale from a DiT-style sigma embedder modulates the noisy
    half only, keeping clean-context computation sigma-independent and
    architecturally identical to the baseline. ``DBLOCK_DIT_FIDELITY=1``
    instead follows the reference DiT recipe: shift/scale for both halves
    (sigma-conditioned context processing) and a sigma-conditioned
    shift/scale on the EDM denoised estimate before the vocabulary head.
    Both additions are zero-initialized, so the paired baseline init is
    unchanged at step 0.

Training grades the denoised estimate ``c_skip * z + c_out * h`` through the
shared softcapped vocabulary head with EDM-weighted cross-entropy.
Validation is the honest generative chain: ``num_blocks`` Euler steps from
pure noise, each routed to the block owning the current sigma, with the
final logits scored as ordinary teacher-forced next-token cross-entropy —
directly comparable to the baseline trainer's ``val_loss``/``val_bpb``.

Unsupported baseline ablation axes (GDN2, MLA, MoE, MTP, NextLat, per-head
Muon) are rejected loudly rather than silently ignored.

Requires ``fla-core==0.5.2``. Diagnostic GPT-2-vocab model; not a 16 MB
submission candidate.
"""

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import itertools
import hashlib
import json
import math
import resource
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from pretraining.byte_accounting import (
    ByteCounter,
    read_dataset_manifest,
    require_matching_vocab_size,
    tokenizer_identity,
)
from pretraining.nanogpt_mini.nanogpt_mini_dblock import (
    READ_ONLY_LOGIT,
    TimestepEmbedder,
    block_sigma_range,
    dblock_mask_mod,
    dblock_positions,
    edm_coefficients,
    equal_mass_sigma_boundaries,
    inference_sigma_schedule,
    interleave_streams,
    noisy_conv_preactivation,
    normalized_embedding_table,
    partition_layers_by_dense,
    sample_block_sigmas,
    sigma_to_block,
)

try:
    import fla
    from fla.modules import FusedRMSNormGated
    from fla.ops.kda import chunk_kda
except ImportError as exc:
    raise RuntimeError(
        "This KDA DiffusionBlocks trainer requires fla-core==0.5.2. "
        "Install it in an isolated environment before launching."
    ) from exc

VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "50304"))
if VOCAB_SIZE % 128:
    raise ValueError(
        f"VOCAB_SIZE={VOCAB_SIZE} must be a multiple of 128; pad the "
        "tokenizer's real vocabulary size up"
    )
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "24"))
if NUM_LAYERS <= 0:
    raise ValueError(f"NUM_LAYERS must be positive, got {NUM_LAYERS}")
default_delta_layer_indices = ",".join(
    str(layer_idx)
    for layer_idx in range(NUM_LAYERS)
    if (layer_idx + 1) % 4
)
DELTA_LAYER_INDICES = frozenset(
    int(index)
    for index in os.environ.get(
        "DELTA_LAYER_INDICES",
        default_delta_layer_indices,
    ).split(",")
    if index
)
if not DELTA_LAYER_INDICES.issubset(range(NUM_LAYERS)):
    raise ValueError(
        f"DELTA_LAYER_INDICES must be within [0, {NUM_LAYERS}), "
        f"got {sorted(DELTA_LAYER_INDICES)}"
    )
KDA_NUM_HEADS = int(os.environ.get("KDA_NUM_HEADS", "3"))
if KDA_NUM_HEADS <= 0:
    raise ValueError(f"KDA_NUM_HEADS must be positive, got {KDA_NUM_HEADS}")
KDA_FULL_RANK_GATE = os.environ.get("KDA_FULL_RANK_GATE", "0") == "1"
NOPE = os.environ.get("NOPE", "0") == "1"
DELTA_DISABLE_RECOMPUTE = (
    os.environ.get(
        "DELTA_DISABLE_RECOMPUTE",
        os.environ.get("KDA_DISABLE_RECOMPUTE", "0"),
    )
    == "1"
)
DELTA_STATE_V_FIRST = os.environ.get("DELTA_STATE_V_FIRST", "1") == "1"
DELTA_BLOCKWISE_COMPILE = (
    os.environ.get("DELTA_BLOCKWISE_COMPILE", "1") == "1"
)
if not DELTA_BLOCKWISE_COMPILE:
    raise NotImplementedError(
        "DELTA_BLOCKWISE_COMPILE=0 has no whole-model compile target here: "
        "DBlockGPT has no forward(), so Module.compile would silently leave "
        "every entry point eager. Per-block compile is the only supported mode."
    )
DELTA_COMPILE_MODE = os.environ.get("DELTA_COMPILE_MODE", "default")
if DELTA_COMPILE_MODE not in {
    "default",
    "reduce-overhead",
    "max-autotune-no-cudagraphs",
    "max-autotune",
}:
    raise ValueError(
        "DELTA_COMPILE_MODE must be 'default', 'reduce-overhead', "
        "'max-autotune-no-cudagraphs', or 'max-autotune', "
        f"got {DELTA_COMPILE_MODE!r}"
    )
if DELTA_COMPILE_MODE in {"reduce-overhead", "max-autotune"}:
    raise NotImplementedError(
        "CUDA-graph compile modes assume every parameter receives a gradient "
        "each step; DiffusionBlocks trains one block per step"
    )
DELTA_MLP_ON_DELTA = os.environ.get("DELTA_MLP_ON_DELTA", "0") == "1"
MLP_HIDDEN = int(os.environ.get("MLP_HIDDEN", "2048"))
if MLP_HIDDEN <= 0:
    raise ValueError(f"MLP_HIDDEN must be positive, got {MLP_HIDDEN}")

for unsupported, name in (
    (os.environ.get("DELTA_ATTENTION_TYPE", "kda") != "kda", "DELTA_ATTENTION_TYPE"),
    (os.environ.get("DENSE_ATTENTION_TYPE", "mha") != "mha", "DENSE_ATTENTION_TYPE"),
    (int(os.environ.get("MOE_NUM_EXPERTS", "0")) != 0, "MOE_NUM_EXPERTS"),
    (int(os.environ.get("MTP_NUM_HEADS", "0")) != 0, "MTP_NUM_HEADS"),
    (os.environ.get("NEXTLAT", "0") == "1", "NEXTLAT"),
    (os.environ.get("PER_HEAD_MUON", "0") == "1", "PER_HEAD_MUON"),
):
    if unsupported:
        raise NotImplementedError(
            f"{name} is not supported by the DiffusionBlocks trainer; run the "
            "baseline trainer for that ablation axis"
        )

########################################
#         DiffusionBlocks config       #
########################################

DBLOCK_GAMMA = float(os.environ.get("DBLOCK_GAMMA", "0.1"))
DBLOCK_SIGMA_MIN = float(os.environ.get("DBLOCK_SIGMA_MIN", "0.002"))
DBLOCK_SIGMA_MAX = float(os.environ.get("DBLOCK_SIGMA_MAX", "80.0"))
DBLOCK_P_MEAN = float(os.environ.get("DBLOCK_P_MEAN", "-1.2"))
DBLOCK_P_STD = float(os.environ.get("DBLOCK_P_STD", "1.2"))
DBLOCK_SIGMA_DATA = float(os.environ.get("DBLOCK_SIGMA_DATA", "0.5"))
if DBLOCK_GAMMA < 0:
    raise ValueError(f"DBLOCK_GAMMA must be nonnegative, got {DBLOCK_GAMMA}")
if not 0 < DBLOCK_SIGMA_MIN < DBLOCK_SIGMA_MAX:
    raise ValueError(
        "need 0 < DBLOCK_SIGMA_MIN < DBLOCK_SIGMA_MAX, got "
        f"{DBLOCK_SIGMA_MIN}, {DBLOCK_SIGMA_MAX}"
    )
if DBLOCK_P_STD <= 0 or DBLOCK_SIGMA_DATA <= 0:
    raise ValueError("DBLOCK_P_STD and DBLOCK_SIGMA_DATA must be positive")

_requested_blocks = int(os.environ.get("DBLOCK_NUM_BLOCKS", "0"))
BLOCK_LAYERS = partition_layers_by_dense(
    DELTA_LAYER_INDICES,
    NUM_LAYERS,
    num_blocks=_requested_blocks if _requested_blocks else None,
)
NUM_BLOCKS = len(BLOCK_LAYERS)
if NUM_BLOCKS < 2:
    raise ValueError(
        "DiffusionBlocks needs at least two blocks; this schedule has "
        f"{NUM_BLOCKS}"
    )
SIGMA_BOUNDARIES = equal_mass_sigma_boundaries(
    NUM_BLOCKS,
    sigma_min=DBLOCK_SIGMA_MIN,
    sigma_max=DBLOCK_SIGMA_MAX,
    p_mean=DBLOCK_P_MEAN,
    p_std=DBLOCK_P_STD,
)
BLOCK_SIGMA_RANGES = [
    block_sigma_range(SIGMA_BOUNDARIES, block_index, DBLOCK_GAMMA)
    for block_index in range(NUM_BLOCKS)
]
DBLOCK_INFER_STEPS = int(
    os.environ.get("DBLOCK_INFER_STEPS", str(NUM_BLOCKS))
)
if DBLOCK_INFER_STEPS < 2:
    raise ValueError(
        f"DBLOCK_INFER_STEPS must be at least 2, got {DBLOCK_INFER_STEPS}"
    )
INFERENCE_SIGMAS = inference_sigma_schedule(
    DBLOCK_INFER_STEPS,
    sigma_min=DBLOCK_SIGMA_MIN,
    sigma_max=DBLOCK_SIGMA_MAX,
    p_mean=DBLOCK_P_MEAN,
    p_std=DBLOCK_P_STD,
)
DBLOCK_VAL_SEED = int(os.environ.get("DBLOCK_VAL_SEED", "271828"))
# Diagnostic probes for the train/inference input-distribution gap; run only
# at the final validation (intended for EVAL_CHECKPOINT jobs).
DBLOCK_DIAG = os.environ.get("DBLOCK_DIAG", "0") == "1"
DBLOCK_DIAG_FINE_STEPS = int(os.environ.get("DBLOCK_DIAG_FINE_STEPS", "24"))
if DBLOCK_DIAG and DBLOCK_DIAG_FINE_STEPS < 2:
    raise ValueError(
        "DBLOCK_DIAG_FINE_STEPS must be at least 2, got "
        f"{DBLOCK_DIAG_FINE_STEPS}"
    )
# Each diagnostic noise stream gets its own 2**33-row seed span so no probe
# can collide with the standard chain's [seed, seed + panel_rows) span for
# any realistic panel size.
DBLOCK_DIAG_SEED_STRIDE = 2**33
# Reference-DiT fidelity: modulate both streams per layer and sigma-condition
# the output head (the reference ViT/DiT recipe the paper's AR models use).
# Off by default; checkpoints record the flag and resume validates it.
DBLOCK_DIT_FIDELITY = os.environ.get("DBLOCK_DIT_FIDELITY", "0") == "1"
# Sigma-conditioned output head alone (0.5M params), without the per-layer
# clean-stream widening. Useful with chain training, where each layer sees a
# single fixed sigma (making per-layer conditioning degenerate) but the
# shared head still spans all six levels' very different denoised statistics.
DBLOCK_COND_HEAD = os.environ.get("DBLOCK_COND_HEAD", "0") == "1"
# Chain-consistent training: instead of z = y + sigma*eps (which leaks the
# target and never shows a block the inputs the inference chain produces),
# Euler-propagate fresh noise through the preceding levels with the current
# weights under no_grad and train the sampled block on that exact chain
# state at its fixed inference sigma, with plain CE (the EDM weight is a
# per-block constant here and is absorbed by the per-block optimizers).
# This removes the mlq-3102-diagnosed train/inference input mismatch by
# construction; gradients still touch only one block plus shared modules.
# "1": one sampled block per step trains on a no-grad prefix rollout.
# "all": fused cascade — one rollout per microbatch trains every block in
# level order; block b's logits are graded (grad, immediate backward) and
# then reused, detached, for the Euler step to level b+1. Only one block's
# autograd graph is alive at a time, so activation memory stays one-block
# scale, while every block sees every batch like the baseline does.
DBLOCK_CHAIN_TRAIN = os.environ.get("DBLOCK_CHAIN_TRAIN", "0")
if DBLOCK_CHAIN_TRAIN not in ("0", "1", "all"):
    raise ValueError(
        f"DBLOCK_CHAIN_TRAIN must be 0, 1, or all, got {DBLOCK_CHAIN_TRAIN}"
    )
CHAIN_TRAIN_SINGLE = DBLOCK_CHAIN_TRAIN == "1"
CHAIN_TRAIN_ALL = DBLOCK_CHAIN_TRAIN == "all"
if DBLOCK_CHAIN_TRAIN != "0":
    if DBLOCK_INFER_STEPS != NUM_BLOCKS:
        raise ValueError(
            "DBLOCK_CHAIN_TRAIN requires DBLOCK_INFER_STEPS == "
            f"DBLOCK_NUM_BLOCKS (got {DBLOCK_INFER_STEPS} vs {NUM_BLOCKS}): "
            "the per-step block sample doubles as the trained chain level"
        )
    for _level, _sigma in enumerate(INFERENCE_SIGMAS):
        if sigma_to_block(_sigma, SIGMA_BOUNDARIES) != _level:
            raise ValueError(
                f"inference level {_level} (sigma {_sigma}) does not route "
                "to its own block; chain training assumes a 1:1 mapping"
            )
# Clean-stream propagation: thread each block's clean-half residual output
# into the next block (detached between blocks), so clean context accumulates
# the full trunk depth across the chain — exactly the baseline's clean
# forward — instead of every block restarting from the raw embedding. The
# clean trajectory is z- and sigma-independent by construction (noisy KDA
# slots are read-only, dense clean queries never see noisy keys, AdaLN
# modulates the noisy half only), so propagation changes which activations a
# block conditions on, not the blockwise gradient structure.
DBLOCK_CLEAN_PROP = os.environ.get("DBLOCK_CLEAN_PROP", "0") == "1"
# Learnable per-level scalar added to c_out before the vocabulary head. At
# the final chain sigma (0.002) the EDM parameterization gives the block's
# hidden state a fixed 0.002 weight against c_skip ~ 1 on z, so the head is
# structurally unable to read what the final block computed (the mlq-3113
# residual final-level gap). Zero-initialized: step-0 behaviour is unchanged.
DBLOCK_READOUT_GAIN = os.environ.get("DBLOCK_READOUT_GAIN", "0") == "1"
if DBLOCK_CLEAN_PROP and not CHAIN_TRAIN_ALL:
    raise NotImplementedError(
        "DBLOCK_CLEAN_PROP=1 requires DBLOCK_CHAIN_TRAIN=all: leak and "
        "single-block training have no in-order block sweep to thread the "
        "clean residual through, and a clean-only prefix path is not "
        "implemented"
    )
if DBLOCK_CLEAN_PROP and DBLOCK_DIT_FIDELITY:
    raise ValueError(
        "DBLOCK_CLEAN_PROP=1 is incompatible with DBLOCK_DIT_FIDELITY=1: "
        "fidelity mode modulates the clean half per sigma, so a propagated "
        "clean residual would depend on the sigma history instead of being "
        "the baseline-identical clean trunk"
    )


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    expected_bytes = 256 * 4 + 2 * num_tokens
    assert file.stat().st_size == expected_bytes, "token shard size does not match header"
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        destination = memoryview(tokens.numpy()).cast("B")
        copied = 0
        # A single readinto larger than INT_MAX is truncated on some libc/
        # Python combinations. Chunk the direct pinned-buffer transfer without
        # introducing Python token lists or an intermediate bytes allocation.
        while copied < len(destination):
            end = min(copied + (1 << 30), len(destination))
            nbytes = f.readinto(destination[copied:end])
            if not nbytes:
                break
            copied += nbytes
        assert copied == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def _data_shard_num_tokens(file: Path) -> int:
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    return int(header[2])


def distributed_data_generator(
    filename_pattern: str,
    batch_size: int,
    seq_len=1024,
    start_step: int = 0,
):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert files, f"no shards match {filename_pattern}"
    assert batch_size % dist.get_world_size() == 0
    if start_step < 0:
        raise ValueError(f"start_step must be nonnegative, got {start_step}")
    local_batch_size = batch_size // dist.get_world_size()
    file_steps = [
        (file, (_data_shard_num_tokens(file) - 1) // batch_size)
        for file in files
    ]
    file_steps = [(file, steps) for file, steps in file_steps if steps > 0]
    files = [file for file, _ in file_steps]
    shard_steps = [steps for _, steps in file_steps]
    steps_per_cycle = sum(shard_steps)
    if steps_per_cycle <= 0:
        raise ValueError("dataset has no complete global batches")
    remaining = start_step % steps_per_cycle
    file_index = 0
    while remaining >= shard_steps[file_index]:
        remaining -= shard_steps[file_index]
        file_index += 1
    ordered_files = files[file_index:] + files[:file_index]
    file_iter = itertools.cycle(ordered_files)
    tokens, pos = _load_data_shard(next(file_iter)), remaining * batch_size
    while True:
        if pos + batch_size + 1 > len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (norm(x.float()) * self.gains).type_as(x)

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class BiasFreeLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor, positions: Tensor):
        # Explicit positions: the concatenated two-stream layout places the
        # noisy token for target ``i`` at absolute position ``i + 1``.
        theta = torch.outer(positions, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    """Two-stream dense attention over the ``[clean(T), noisy(T)]`` layout.

    flex_attention with the DiffusionBlocks mask replaces the baseline's
    ``is_causal`` SDPA: the clean half is ordinarily causal (identical math
    to the baseline), the noisy half cross-attends to its clean prefix plus
    itself. Scale 0.12 is the baseline's.
    """

    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor, block_mask, positions: Tensor):
        B, T2 = x.size(0), x.size(1)
        q = self.q(x).view(B, T2, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T2, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T2, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        if not NOPE:
            q, k = self.rotary(q, positions), self.rotary(k, positions)
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            scale=0.12,
        ).transpose(1, 2)
        y = y.contiguous().view(B, T2, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y


class ShortConv(nn.Module):
    """Depthwise causal conv + SiLU with FLA's ``[D, 1, W]`` ``weight`` so
    checkpoints stay key-compatible with ``ShortConvolution``."""

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))

    def two_stream(self, clean: Tensor, noisy: Tensor) -> tuple[Tensor, Tensor]:
        """SiLU-activated clean and noisy branches over ``[B, T, D]`` inputs.

        The clean branch is the ordinary causal conv. The noisy branch's
        window replaces the newest tap with the noisy projection: position
        ``i`` sees ``[c_{i-W+2}, ..., c_i, n_i]`` — the window the original
        network would see were the next input the noisy embedding.
        """
        clean_pre = F.conv1d(
            clean.transpose(1, 2),
            self.weight.type_as(clean),
            groups=self.hidden_size,
            padding=self.kernel_size - 1,
        )[..., : clean.size(1)].transpose(1, 2)
        noisy_pre = noisy_conv_preactivation(clean, noisy, self.weight)
        return F.silu(clean_pre), F.silu(noisy_pre)


class KimiDeltaAttention(nn.Module):
    """Two-stream KDA mixer over the concatenated layout.

    Interleaves the streams ``[c_0, n_0, c_1, n_1, ...]`` through the
    released chunk kernel; noisy slots pass ``READ_ONLY_LOGIT`` decay and
    beta logits, which the in-kernel sigmoids turn into decay 1 and write
    strength 0. The clean state trajectory is therefore exactly the
    clean-only trajectory and every noisy slot reads the state after its
    clean prefix. Decay/beta projections only ever run on the clean half, so
    they receive no spurious gradient from overridden logits.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 128,
        num_heads: int = KDA_NUM_HEADS,
        conv_size: int = 4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = self.num_heads * head_dim

        self.q_proj = BiasFreeLinear(dim, self.projection_size)
        self.k_proj = BiasFreeLinear(dim, self.projection_size)
        self.v_proj = BiasFreeLinear(dim, self.projection_size)
        self.q_conv1d = ShortConv(self.projection_size, conv_size)
        self.k_conv1d = ShortConv(self.projection_size, conv_size)
        self.v_conv1d = ShortConv(self.projection_size, conv_size)

        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.f_a_proj = BiasFreeLinear(dim, head_dim)
        self.f_b_proj = BiasFreeLinear(head_dim, self.projection_size)
        self.dt_bias = nn.Parameter(torch.empty(self.projection_size, dtype=torch.float32))
        self.b_proj = BiasFreeLinear(dim, self.num_heads)
        if KDA_FULL_RANK_GATE:
            self.g_proj = BiasFreeLinear(dim, self.projection_size)
        else:
            self.g_a_proj = BiasFreeLinear(dim, head_dim)
            self.g_b_proj = BiasFreeLinear(head_dim, self.projection_size)
        self.o_norm = FusedRMSNormGated(head_dim, eps=1e-6, activation="sigmoid")
        self.o_proj = BiasFreeLinear(self.projection_size, dim)

    def forward(self, x: Tensor):
        B, T2, _ = x.shape
        T = T2 // 2
        clean_x, noisy_x = x[:, :T], x[:, T:]
        heads = (B, T, self.num_heads, self.head_dim)

        q_all = self.q_proj(x)
        k_all = self.k_proj(x)
        v_all = self.v_proj(x)
        q_c, q_n = self.q_conv1d.two_stream(q_all[:, :T], q_all[:, T:])
        k_c, k_n = self.k_conv1d.two_stream(k_all[:, :T], k_all[:, T:])
        v_c, v_n = self.v_conv1d.two_stream(v_all[:, :T], v_all[:, T:])

        decay_clean = self.f_b_proj(self.f_a_proj(clean_x)).view(*heads)
        decay_noisy = torch.full_like(decay_clean, READ_ONLY_LOGIT)
        beta_clean = self.b_proj(clean_x).float()
        beta_noisy = torch.full_like(beta_clean, READ_ONLY_LOGIT)

        q_int = interleave_streams(q_c.view(*heads), q_n.view(*heads))
        k_int = interleave_streams(k_c.view(*heads), k_n.view(*heads))
        v_int = interleave_streams(v_c.view(*heads), v_n.view(*heads))
        g_int = interleave_streams(decay_clean, decay_noisy)
        beta_int = interleave_streams(beta_clean, beta_noisy)

        y_int, _ = chunk_kda(
            q=q_int,
            k=k_int,
            v=v_int,
            g=g_int,
            beta=beta_int,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            state_v_first=DELTA_STATE_V_FIRST,
            disable_recompute=DELTA_DISABLE_RECOMPUTE,
        )
        y = torch.cat((y_int[:, 0::2], y_int[:, 1::2]), dim=1)
        output_gate = (
            self.g_proj(x)
            if KDA_FULL_RANK_GATE
            else self.g_b_proj(self.g_a_proj(x))
        ).view(B, T2, self.num_heads, self.head_dim)
        y = self.o_norm(y, output_gate).reshape(B, T2, self.projection_size)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = Linear(dim, MLP_HIDDEN)
        self.proj = Linear(MLP_HIDDEN, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    """Baseline residual block plus zero-initialized AdaLN shift/scale.

    Default: modulate only the noisy half, so the clean context path stays
    sigma-independent and identical to the baseline architecture. With
    ``DBLOCK_DIT_FIDELITY`` the heads emit clean and noisy shift/scale pairs
    and both halves are modulated, matching the reference DiT layers that
    condition every token on sigma; zero init keeps step-0 behaviour equal
    to the baseline either way."""

    def __init__(self, dim: int, use_kda: bool, cond_dim: int):
        super().__init__()
        self.use_kda = use_kda
        if use_kda:
            self.attn = KimiDeltaAttention(dim)
        else:
            self.attn = CausalSelfAttention(dim)
        mod_width = (4 if DBLOCK_DIT_FIDELITY else 2) * dim
        self.norm1 = RMSNorm(dim)
        self.ada_attn = BiasFreeLinear(cond_dim, mod_width)
        self.use_mlp = not use_kda or DELTA_MLP_ON_DELTA
        if self.use_mlp:
            self.mlp = MLP(dim)
            self.norm2 = RMSNorm(dim)
            self.ada_mlp = BiasFreeLinear(cond_dim, mod_width)

    @staticmethod
    def _modulate(h: Tensor, mod: Tensor) -> Tensor:
        T = h.size(1) // 2
        mod = mod.type_as(h)
        if DBLOCK_DIT_FIDELITY:
            c_shift, c_scale, shift, scale = mod.chunk(4, dim=-1)
            clean = h[:, :T] * (1 + c_scale[:, None, :]) + c_shift[:, None, :]
        else:
            shift, scale = mod.chunk(2, dim=-1)
            clean = h[:, :T]
        noisy = h[:, T:] * (1 + scale[:, None, :]) + shift[:, None, :]
        return torch.cat((clean, noisy), dim=1)

    def forward(self, x: Tensor, cond: Tensor, block_mask, positions: Tensor):
        h = self._modulate(self.norm1(x), self.ada_attn(cond))
        if self.use_kda:
            x = x + self.attn(h)
        else:
            x = x + self.attn(h, block_mask, positions)
        if self.use_mlp:
            h = self._modulate(self.norm2(x), self.ada_mlp(cond))
            x = x + self.mlp(h)
        return x


class DBlockGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            Block(
                model_dim,
                use_kda=layer_idx in DELTA_LAYER_INDICES,
                cond_dim=model_dim,
            )
            for layer_idx in range(num_layers)
        ])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.time_embed = TimestepEmbedder(model_dim)
        # Reference forward_output_embeddings: sigma-conditioned shift/scale
        # on the denoised estimate before the classifier, zero-initialized.
        self.head_ada = (
            BiasFreeLinear(model_dim, 2 * model_dim)
            if (DBLOCK_DIT_FIDELITY or DBLOCK_COND_HEAD)
            else None
        )
        # Learnable additive readout: logits read c_skip*z + (c_out + g_b)*h,
        # so a low-sigma level is not pinned to c_out ~ sigma visibility of
        # its own computation. One scalar per level, zero-initialized.
        self.readout_gain = (
            nn.Parameter(torch.zeros(NUM_BLOCKS, dtype=torch.float32))
            if DBLOCK_READOUT_GAIN
            else None
        )
        # 0-d stand-in so the compiled head sees one graph either way; not a
        # persistent buffer, so disabled-mode checkpoints are unchanged.
        self.register_buffer(
            "_readout_zero", torch.zeros(()), persistent=False
        )
        self.block_layers = BLOCK_LAYERS

    def _readout_gain(self, block_index: int) -> Tensor:
        """Per-level learnable readout gain (0-d tensor); zero when disabled.

        Indexing happens outside the compiled head so one compiled graph
        serves every level while the gradient still reaches the parameter."""
        if self.readout_gain is not None:
            return self.readout_gain[block_index]
        return self._readout_zero

    def _run_block(
        self,
        inputs: Tensor,
        z_sigma: Tensor,
        coeffs: dict[str, Tensor],
        block_index: int,
        block_mask,
        positions: Tensor,
        clean_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """(noisy-half, clean-half) hidden states after one block's layers.

        ``clean_in`` (DBLOCK_CLEAN_PROP) is the previous block's clean-half
        residual output, so clean context accumulates the baseline's full
        trunk depth across the chain instead of restarting from the
        embedding at every block. Callers detach it between levels, keeping
        gradients blockwise."""
        clean = self.norm1(self.embed(inputs)) if clean_in is None else clean_in
        noisy = (coeffs["c_in"][:, None, None] * z_sigma).to(clean.dtype)
        x = torch.cat((clean, noisy), dim=1)
        cond = self.time_embed(coeffs["c_noise"]).to(clean.dtype)
        for layer_idx in self.block_layers[block_index]:
            x = self.blocks[layer_idx](x, cond, block_mask, positions)
        T = inputs.size(1)
        return x[:, T:], x[:, :T]

    def _edm_head(
        self,
        hidden_noisy: Tensor,
        z_sigma: Tensor,
        c_skip: Tensor,
        c_out: Tensor,
        c_noise: Tensor,
        gain: Tensor,
    ) -> Tensor:
        """Softcapped vocabulary logits from the EDM denoised estimate.

        ``gain`` is the level's learnable additive readout scale on the
        hidden term (0 when DBLOCK_READOUT_GAIN is off), freeing low-sigma
        levels from the c_out ~ sigma visibility crush."""
        hidden = self.norm2(hidden_noisy)
        denoised = (
            c_skip[:, None, None] * z_sigma
            + (c_out[:, None, None] + gain.float()) * hidden.float()
        )
        if self.head_ada is not None:
            cond = self.time_embed(c_noise).to(hidden_noisy.dtype)
            shift, scale = self.head_ada(cond).float().chunk(2, dim=-1)
            denoised = (
                denoised * (1 + scale[:, None, :]) + shift[:, None, :]
            )
        logits = self.proj(denoised.to(hidden_noisy.dtype)).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()

    # ``head_logits``/``loss_tail`` are rebound to compiled versions at setup
    # (the baseline's compute_loss pattern); block layers compile separately.
    def head_logits(
        self,
        hidden_noisy: Tensor,
        z_sigma: Tensor,
        c_skip: Tensor,
        c_out: Tensor,
        c_noise: Tensor,
        gain: Tensor,
    ) -> Tensor:
        return self._edm_head(
            hidden_noisy, z_sigma, c_skip, c_out, c_noise, gain
        )

    def loss_tail(
        self,
        hidden_noisy: Tensor,
        z_sigma: Tensor,
        c_skip: Tensor,
        c_out: Tensor,
        c_noise: Tensor,
        gain: Tensor,
        targets: Tensor,
        weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = self._edm_head(
            hidden_noisy, z_sigma, c_skip, c_out, c_noise, gain
        )
        ce = F.cross_entropy(
            logits.view(targets.numel(), -1),
            targets.view(-1),
            reduction="none",
        ).view_as(targets)
        return (ce * weight[:, None]).sum(), ce.detach().sum()

    def loss_tail_chain(
        self,
        hidden_noisy: Tensor,
        z_sigma: Tensor,
        c_skip: Tensor,
        c_out: Tensor,
        c_noise: Tensor,
        gain: Tensor,
        targets: Tensor,
        table: Tensor,
        need_denoised: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Fused-cascade tail: plain-CE sum for the level plus (except at
        the final level) the detached posterior-expectation embedding for
        the next Euler step, sharing one compiled softmax pipeline instead
        of an eager fp32 re-pass over the vocabulary. Numerics match
        ``denoised_expectation``: fp32 softmax, bf16 tensor-core
        contraction, fp32 result."""
        logits = self._edm_head(
            hidden_noisy, z_sigma, c_skip, c_out, c_noise, gain
        )
        ce = F.cross_entropy(
            logits.view(targets.numel(), -1),
            targets.view(-1),
            reduction="sum",
        )
        if not need_denoised:
            return ce, None
        probs = logits.detach().softmax(-1).to(torch.bfloat16)
        denoised = (probs @ table.to(torch.bfloat16)).float()
        return ce, denoised

    def dblock_cascade_level(
        self,
        inputs: Tensor,
        targets: Tensor,
        block_index: int,
        z_sigma: Tensor,
        sigma_rows: Tensor,
        block_mask,
        positions: Tensor,
        table: Tensor,
        need_denoised: bool,
        clean_in: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """One fused-cascade training level (see CHAIN_TRAIN_ALL): gradient
        CE for ``block_index`` on the chain state ``z_sigma`` plus the
        detached expectation feeding the next level, plus (under
        DBLOCK_CLEAN_PROP) the detached clean residual for the next block."""
        coeffs = edm_coefficients(sigma_rows, DBLOCK_SIGMA_DATA)
        hidden_noisy, clean_out = self._run_block(
            inputs, z_sigma, coeffs, block_index, block_mask, positions,
            clean_in,
        )
        ce, denoised = self.loss_tail_chain(
            hidden_noisy,
            z_sigma,
            coeffs["c_skip"],
            coeffs["c_out"],
            coeffs["c_noise"],
            self._readout_gain(block_index),
            targets,
            table,
            need_denoised,
        )
        return ce, denoised, (clean_out.detach() if DBLOCK_CLEAN_PROP else None)

    def denoise_logits(
        self,
        inputs: Tensor,
        z_sigma: Tensor,
        sigma_rows: Tensor,
        block_index: int,
        block_mask,
        positions: Tensor,
        clean_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """(softcapped logits, clean-half residual) for one diffusion block
        (validation chain). Chain callers under DBLOCK_CLEAN_PROP thread the
        returned clean residual into the next level; others ignore it."""
        coeffs = edm_coefficients(sigma_rows, DBLOCK_SIGMA_DATA)
        hidden_noisy, clean_out = self._run_block(
            inputs, z_sigma, coeffs, block_index, block_mask, positions,
            clean_in,
        )
        logits = self.head_logits(
            hidden_noisy,
            z_sigma,
            coeffs["c_skip"],
            coeffs["c_out"],
            coeffs["c_noise"],
            self._readout_gain(block_index),
        )
        return logits, clean_out

    def dblock_loss(
        self,
        inputs: Tensor,
        targets: Tensor,
        block_index: int,
        sigma_rows: Tensor,
        block_mask,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """(EDM-weighted CE sum, raw CE sum) for one sampled block."""
        table = normalized_embedding_table(self.embed.weight)
        y = table[targets]
        z_sigma = y + sigma_rows[:, None, None] * torch.randn_like(y)
        coeffs = edm_coefficients(sigma_rows, DBLOCK_SIGMA_DATA)
        hidden_noisy, _ = self._run_block(
            inputs, z_sigma, coeffs, block_index, block_mask, positions
        )
        return self.loss_tail(
            hidden_noisy,
            z_sigma,
            coeffs["c_skip"],
            coeffs["c_out"],
            coeffs["c_noise"],
            self._readout_gain(block_index),
            targets,
            coeffs["weight"],
        )

    def dblock_chain_loss(
        self,
        inputs: Tensor,
        targets: Tensor,
        block_index: int,
        z_sigma: Tensor,
        sigma_rows: Tensor,
        block_mask,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """(CE sum, CE sum) for one block on a chain-propagated noisy input.

        Unweighted: with chain training every block trains at its single
        fixed inference sigma, so the EDM weight is a per-block constant the
        per-block Adam/Muon optimizers absorb."""
        coeffs = edm_coefficients(sigma_rows, DBLOCK_SIGMA_DATA)
        hidden_noisy, _ = self._run_block(
            inputs, z_sigma, coeffs, block_index, block_mask, positions
        )
        return self.loss_tail(
            hidden_noisy,
            z_sigma,
            coeffs["c_skip"],
            coeffs["c_out"],
            coeffs["c_noise"],
            self._readout_gain(block_index),
            targets,
            torch.ones_like(sigma_rows),
        )


def denoised_expectation(logits: Tensor, table: Tensor) -> Tensor:
    """``E_p[y]`` under the predicted token distribution, row-chunked so the
    full-width probability tensor never materializes. The softmax runs in
    fp32; the vocab-wide contraction runs on bf16 tensor cores (probabilities
    are non-negative and sum to one, so bf16's 8 relative bits bound the
    expectation error at ~2^-9 of the unit-norm embedding scale — far below
    the sigma of any chain level that consumes it)."""
    out = torch.empty(
        *logits.shape[:-1], table.size(1), dtype=torch.float32,
        device=logits.device,
    )
    table_bf16 = table.to(torch.bfloat16)
    for start in range(0, logits.size(0), 4):
        stop = min(start + 4, logits.size(0))
        probs = logits[start:stop].softmax(-1).to(torch.bfloat16)
        out[start:stop] = (probs @ table_bf16).float()
    return out


@torch.no_grad()
def chain_validation_loss(
    model: DBlockGPT,
    inputs: Tensor,
    targets: Tensor,
    block_mask,
    positions: Tensor,
    row_offset: int,
) -> tuple[Tensor, list[Tensor]]:
    """Teacher-forced generative chain: ``DBLOCK_INFER_STEPS`` Euler steps
    from pure noise, each routed to the block owning the current sigma. The
    final logits are graded as ordinary next-token CE (sum), directly
    comparable to the baseline trainer's validation loss. Also returns the
    per-level CE sums for refinement diagnostics.

    The starting noise is seeded per absolute panel row (per rank), so the
    reported loss is invariant to MBS — matching the baseline's contract
    that MBS may vary across a resume without moving validation numbers."""
    table = normalized_embedding_table(model.embed.weight)
    z = torch.empty(
        *targets.shape, model.model_dim,
        dtype=torch.float32, device=inputs.device,
    )
    for row in range(targets.size(0)):
        row_generator = torch.Generator(device=inputs.device)
        row_generator.manual_seed(DBLOCK_VAL_SEED + row_offset + row)
        torch.randn(
            targets.size(1), model.model_dim,
            dtype=torch.float32, device=inputs.device,
            generator=row_generator, out=z[row],
        )
    z *= INFERENCE_SIGMAS[0]
    per_level = []
    clean = None
    for index, sigma in enumerate(INFERENCE_SIGMAS):
        block_index = sigma_to_block(sigma, SIGMA_BOUNDARIES)
        sigma_rows = torch.full(
            (inputs.size(0),), sigma, dtype=torch.float32, device=inputs.device
        )
        logits, clean_out = model.denoise_logits(
            inputs, z, sigma_rows, block_index, block_mask, positions, clean
        )
        if DBLOCK_CLEAN_PROP:
            clean = clean_out
        per_level.append(
            F.cross_entropy(
                logits.view(targets.numel(), -1),
                targets.view(-1),
                reduction="sum",
            )
        )
        if index + 1 < len(INFERENCE_SIGMAS):
            denoised = denoised_expectation(logits, table)
            slope = (z - denoised) / sigma
            z = z + (INFERENCE_SIGMAS[index + 1] - sigma) * slope
    return per_level[-1], per_level


@torch.no_grad()
def chain_training_input(
    model: DBlockGPT,
    inputs: Tensor,
    block_index: int,
    block_mask,
    positions: Tensor,
) -> Tensor:
    """Euler-propagate fresh noise through inference levels
    ``0..block_index-1`` with the current weights, yielding exactly the
    ``z`` the validation chain would hand this block. Fresh noise draws come
    from the global training RNG (restored on resume). The returned tensor
    carries no gradient history, so a subsequent loss backpropagates only
    through the trained block and the shared head/embedding."""
    table = normalized_embedding_table(model.embed.weight)
    z = (
        torch.randn(
            inputs.size(0), inputs.size(1), model.model_dim,
            dtype=torch.float32, device=inputs.device,
        )
        * INFERENCE_SIGMAS[0]
    )
    for level in range(block_index):
        sigma = INFERENCE_SIGMAS[level]
        level_block = sigma_to_block(sigma, SIGMA_BOUNDARIES)
        sigma_rows = torch.full(
            (inputs.size(0),), sigma, dtype=torch.float32, device=inputs.device
        )
        logits, _ = model.denoise_logits(
            inputs, z, sigma_rows, level_block, block_mask, positions
        )
        denoised = denoised_expectation(logits, table)
        slope = (z - denoised) / sigma
        z = z + (INFERENCE_SIGMAS[level + 1] - sigma) * slope
    return z


@torch.no_grad()
def oracle_level_losses(
    model: DBlockGPT,
    inputs: Tensor,
    targets: Tensor,
    block_mask,
    positions: Tensor,
    row_offset: int,
) -> list[Tensor]:
    """CE of each inference level's block on its *training* input
    distribution ``z = y + sigma * eps`` at the exact chain sigma. Separates
    "a block is mis-wired" from "the chain feeds it out-of-distribution
    estimates": oracle CE near the block's training CE clears the wiring."""
    table = normalized_embedding_table(model.embed.weight)
    y = table[targets]
    losses = []
    clean = None
    for index, sigma in enumerate(INFERENCE_SIGMAS):
        block_index = sigma_to_block(sigma, SIGMA_BOUNDARIES)
        eps = torch.empty_like(y)
        for row in range(targets.size(0)):
            row_generator = torch.Generator(device=inputs.device)
            row_generator.manual_seed(
                DBLOCK_VAL_SEED
                + (index + 1) * DBLOCK_DIAG_SEED_STRIDE
                + row_offset
                + row
            )
            torch.randn(
                targets.size(1), model.model_dim,
                dtype=torch.float32, device=inputs.device,
                generator=row_generator, out=eps[row],
            )
        z = y + sigma * eps
        sigma_rows = torch.full(
            (inputs.size(0),), sigma, dtype=torch.float32, device=inputs.device
        )
        # The clean residual is z-independent, so threading it through the
        # per-level oracle passes reproduces the propagated chain's clean
        # inputs exactly even though each level's z here is fresh.
        logits, clean_out = model.denoise_logits(
            inputs, z, sigma_rows, block_index, block_mask, positions, clean
        )
        if DBLOCK_CLEAN_PROP:
            clean = clean_out
        losses.append(
            F.cross_entropy(
                logits.view(targets.numel(), -1),
                targets.view(-1),
                reduction="sum",
            )
        )
    return losses


def _sampled_embedding(
    logits: Tensor, table: Tensor, row_offset: int, level_index: int
) -> Tensor:
    """Embedding of one token sampled per position from the level posterior.
    Seeded and drawn per absolute panel row so the draw is MBS-invariant,
    matching the chain and oracle probes' contract. The per-level stream
    offset (1000 + level) keeps sampling seeds disjoint from the oracle
    probe's (index + 1) spans."""
    tokens = torch.empty(
        logits.shape[:-1], dtype=torch.long, device=logits.device
    )
    for row in range(logits.size(0)):
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(
            DBLOCK_VAL_SEED
            + (1000 + level_index) * DBLOCK_DIAG_SEED_STRIDE
            + row_offset
            + row
        )
        tokens[row] = torch.multinomial(
            logits[row].float().softmax(-1), 1, generator=generator
        ).squeeze(-1)
    return table[tokens]


@torch.no_grad()
def harvest_clean_inputs(
    model: DBlockGPT,
    inputs: Tensor,
    block_mask,
    positions: Tensor,
) -> list[Tensor | None]:
    """Per-block clean-half inputs under DBLOCK_CLEAN_PROP.

    The clean residual is independent of the noisy stream and of sigma, so
    one in-order sweep with a zero noisy stream yields the exact clean input
    every block would see in the propagated chain. Needed by probes whose
    sigma grids revisit or skip blocks (the fine diagnostic chain), where
    in-loop threading is ill-defined. ``[0]`` is ``None``: block 0 computes
    its clean input from the embedding."""
    clean_inputs: list[Tensor | None] = [None] * NUM_BLOCKS
    z = torch.zeros(
        inputs.size(0), inputs.size(1), model.model_dim,
        dtype=torch.float32, device=inputs.device,
    )
    clean = None
    for block_index in range(NUM_BLOCKS):
        clean_inputs[block_index] = clean
        sigma_rows = torch.full(
            (inputs.size(0),),
            INFERENCE_SIGMAS[block_index],
            dtype=torch.float32,
            device=inputs.device,
        )
        coeffs = edm_coefficients(sigma_rows, DBLOCK_SIGMA_DATA)
        _, clean = model._run_block(
            inputs, z, coeffs, block_index, block_mask, positions, clean
        )
    return clean_inputs


@torch.no_grad()
def diagnostic_chain_loss(
    model: DBlockGPT,
    inputs: Tensor,
    targets: Tensor,
    block_mask,
    positions: Tensor,
    row_offset: int,
    sigmas: list[float],
    estimate: str,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Chain variant where ``estimate`` picks how the Euler denoised estimate
    is formed from each level's posterior:

    - "expectation": the standard softmax-weighted mean embedding (with the
      standard grid this must reproduce ``chain_validation_loss`` exactly);
    - "renorm": that expectation rescaled to unit L2 norm, fixing the scale
      collapse of high-entropy posteriors without fixing direction;
    - "sample": the embedding of one sampled token, which stays on the
      unit-norm manifold the blocks were trained on (ancestral rather than
      mean-field; not a likelihood, diagnostic only).

    Returns per-level (CE sums, estimate L2-norm sums, cosine-to-target
    sums); the norm/cos lists have one fewer entry because the final level
    produces no further estimate. The starting noise reuses the standard
    chain's per-row seeds."""
    if estimate not in ("expectation", "renorm", "sample"):
        raise ValueError(f"unknown estimate mode: {estimate}")
    table = normalized_embedding_table(model.embed.weight)
    y_true = table[targets]
    z = torch.empty(
        *targets.shape, model.model_dim,
        dtype=torch.float32, device=inputs.device,
    )
    for row in range(targets.size(0)):
        row_generator = torch.Generator(device=inputs.device)
        row_generator.manual_seed(DBLOCK_VAL_SEED + row_offset + row)
        torch.randn(
            targets.size(1), model.model_dim,
            dtype=torch.float32, device=inputs.device,
            generator=row_generator, out=z[row],
        )
    z *= sigmas[0]
    level_ce: list[Tensor] = []
    level_norm: list[Tensor] = []
    level_cos: list[Tensor] = []
    clean_inputs = (
        harvest_clean_inputs(model, inputs, block_mask, positions)
        if DBLOCK_CLEAN_PROP
        else [None] * NUM_BLOCKS
    )
    for index, sigma in enumerate(sigmas):
        block_index = sigma_to_block(sigma, SIGMA_BOUNDARIES)
        sigma_rows = torch.full(
            (inputs.size(0),), sigma, dtype=torch.float32, device=inputs.device
        )
        logits, _ = model.denoise_logits(
            inputs, z, sigma_rows, block_index, block_mask, positions,
            clean_inputs[block_index],
        )
        level_ce.append(
            F.cross_entropy(
                logits.view(targets.numel(), -1),
                targets.view(-1),
                reduction="sum",
            )
        )
        if index + 1 < len(sigmas):
            if estimate == "sample":
                denoised = _sampled_embedding(logits, table, row_offset, index)
                level_norm.append(denoised.norm(dim=-1).sum())
            else:
                denoised = denoised_expectation(logits, table)
                # Record the raw expectation's norm even in renorm mode;
                # after rescaling it is 1 by construction and carries no
                # information.
                level_norm.append(denoised.norm(dim=-1).sum())
                if estimate == "renorm":
                    denoised = denoised * denoised.square().sum(
                        -1, keepdim=True
                    ).clamp(min=1e-12).rsqrt()
            level_cos.append(
                F.cosine_similarity(denoised.float(), y_true, dim=-1).sum()
            )
            slope = (z - denoised) / sigma
            z = z + (sigmas[index + 1] - sigma) * slope
    return level_ce, level_norm, level_cos


@torch.no_grad()
def initialize_model(model: DBlockGPT, seed: int):
    """Pair the unchanged trunk with the baseline and initialize KDA safely.

    Reuses the baseline trainer's RNG discipline so every parameter the
    baseline architecture also has receives the identical draw; the modules
    DiffusionBlocks adds (AdaLN heads, sigma embedder) use their own seeded
    generators and start inert (zero modulation)."""

    torch.manual_seed(seed)
    for p in model.parameters():
        p.fill_(float("nan"))

    def normal_weight(weight: Tensor, generator=None):
        std = 0.33**0.5 / weight.size(-1)**0.5
        weight.normal_(std=std, generator=generator)

    def paired_prefix_weight(
        weight: Tensor,
        baseline_shape: tuple[int, int],
        reference_generator=None,
        extra_generator=None,
    ):
        reference = weight.new_empty(baseline_shape)
        normal_weight(reference, generator=reference_generator)
        if weight.size(1) > reference.size(1):
            raise ValueError("Paired weight cannot be wider than its baseline")
        paired_rows = min(weight.size(0), reference.size(0))
        weight[:paired_rows].copy_(reference[:paired_rows, :weight.size(1)])
        if paired_rows < weight.size(0):
            if extra_generator is None:
                raise ValueError("Expanded paired weight requires an extra generator")
            normal_weight(weight[paired_rows:], generator=extra_generator)

    def zero_linear(linear: nn.Linear):
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.zero_()

    model.embed.weight.normal_()
    for layer_idx, block in enumerate(model.blocks):
        if isinstance(block.attn, KimiDeltaAttention):
            attn = block.attn
            local_generator = torch.Generator(device=attn.q_proj.weight.device)
            local_generator.manual_seed(seed + 10_000 + layer_idx)

            baseline_attn_shape = (attn.q_proj.in_features, attn.q_proj.in_features)
            paired_prefix_weight(
                attn.q_proj.weight,
                baseline_attn_shape,
                reference_generator=local_generator,
            )
            paired_prefix_weight(
                attn.k_proj.weight,
                baseline_attn_shape,
                reference_generator=local_generator,
            )
            paired_prefix_weight(
                attn.v_proj.weight,
                baseline_attn_shape,
                reference_generator=local_generator,
            )
            gate_linears = (
                (attn.g_proj,)
                if KDA_FULL_RANK_GATE
                else (attn.g_a_proj, attn.g_b_proj)
            )
            for linear in (
                attn.f_a_proj,
                attn.f_b_proj,
                attn.b_proj,
                *gate_linears,
            ):
                normal_weight(linear.weight, generator=local_generator)
                if linear.bias is not None:
                    linear.bias.zero_()

            # An identity causal convolution preserves the current-token
            # projection before the required SiLU at initialization.
            for conv in (attn.q_conv1d, attn.k_conv1d, attn.v_conv1d):
                conv.weight.zero_()
                conv.weight[:, 0, -1] = 1

            # KDA retains its bounded safe-gate initialization at A=1.
            attn.A_log.zero_()
            # Replay the completed KDA reference's pre-dt RNG draws so the
            # decay time constants match the baseline trainer bit-for-bit.
            decay_generator = torch.Generator(device=attn.q_proj.weight.device)
            decay_generator.manual_seed(seed + 10_000 + layer_idx)
            for shape in (
                (attn.head_dim, attn.q_proj.in_features),
                (attn.projection_size, attn.head_dim),
                (KDA_NUM_HEADS, attn.q_proj.in_features),
                (attn.projection_size, attn.q_proj.in_features),
            ):
                reference_draw = attn.q_proj.weight.new_empty(shape)
                normal_weight(reference_draw, generator=decay_generator)
            log_dt = torch.empty_like(attn.dt_bias).uniform_(
                math.log(0.001),
                math.log(0.1),
                generator=decay_generator,
            )
            dt = log_dt.exp().clamp(min=1e-4)
            attn.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            attn.o_norm.weight.fill_(1)
            zero_linear(attn.o_proj)
        else:
            attn = block.attn
            assert isinstance(attn, CausalSelfAttention)
            for linear in (attn.q, attn.k, attn.v):
                normal_weight(linear.weight)
                linear.bias.zero_()
            zero_linear(attn.proj)

        if block.use_kda:
            block.norm1.gains.fill_(1)
        else:
            block.norm1.gains.normal_(mean=1, std=0)
        zero_linear(block.ada_attn)
        if block.use_mlp:
            baseline_mlp_shape = (
                4 * block.mlp.fc.in_features,
                block.mlp.fc.in_features,
            )
            mlp_extra_generator = torch.Generator(
                device=block.mlp.fc.weight.device
            )
            mlp_extra_generator.manual_seed(seed + 30_000 + layer_idx)
            paired_prefix_weight(
                block.mlp.fc.weight,
                baseline_mlp_shape,
                reference_generator=(
                    mlp_extra_generator if block.use_kda else None
                ),
                extra_generator=mlp_extra_generator,
            )
            block.mlp.fc.bias.zero_()
            zero_linear(block.mlp.proj)
            if block.use_kda:
                block.norm2.gains.fill_(1)
            else:
                block.norm2.gains.normal_(mean=1, std=0)
            zero_linear(block.ada_mlp)

    zero_linear(model.proj)
    if model.head_ada is not None:
        zero_linear(model.head_ada)
    if model.readout_gain is not None:
        model.readout_gain.zero_()
    model.norm1.gains.normal_(mean=1, std=0)
    model.norm2.gains.normal_(mean=1, std=0)

    time_generator = torch.Generator(device=model.embed.weight.device)
    time_generator.manual_seed(seed + 60_000)
    for linear in (model.time_embed.fc1, model.time_embed.fc2):
        linear.weight.normal_(std=0.02, generator=time_generator)
        linear.bias.zero_()

    uninitialized = [name for name, p in model.named_parameters() if not p.isfinite().all()]
    if uninitialized:
        raise RuntimeError(f"Uninitialized parameters: {uninitialized}")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Quintic Newton-Schulz iteration from current Muon/modded-nanoGPT. The
    # coefficients intentionally do not converge all the way to the polar
    # factor: the resulting S-shaped singular-value map is both faster and
    # empirically better behaved than the older 12-step cubic iteration.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    if "mu" not in state:
                        # Checkpoints written before mu became tensor-valued
                        # restore momentum without it; recreate on resume.
                        state["mu"] = torch.empty(
                            (), device=p.device, dtype=torch.float32
                        )
                    # Tensor-valued momentum keeps the compiled update generic
                    # across the scalar momentum warmup.
                    state["mu"].fill_(group["mu"])
                    update = muon_update(
                        p.grad,
                        state["momentum"],
                        mu=state["mu"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                if world_size > 1:
                    dist.all_gather(
                        params_pad[base_i:base_i + world_size],
                        params_pad[base_i + rank],
                    )


########################################
#                Setup                 #
########################################

# Single-process shim: default the torchrun env vars so `python3` launches work.
if "RANK" not in os.environ:
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29648")

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
world_size = dist.get_world_size()
assert 8 % world_size == 0

# logging setup
run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)

def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
attention_schedule = ",".join(
    "KDA" if layer_idx in DELTA_LAYER_INDICES else "dense"
    for layer_idx in range(NUM_LAYERS)
)
print0(
    f"Using fla-core {fla.__version__}; attention schedule: "
    f"{attention_schedule}"
)
print0(f"KDA disable_recompute: {DELTA_DISABLE_RECOMPUTE}")
print0(f"KDA state_v_first: {DELTA_STATE_V_FIRST}")
print0(f"KDA blockwise compile: {DELTA_BLOCKWISE_COMPILE}")
print0(f"KDA layers: {sorted(DELTA_LAYER_INDICES)}")
print0(f"KDA heads: {KDA_NUM_HEADS}; full-rank output gate: {KDA_FULL_RANK_GATE}")
print0(f"dense attention: mha; position encoding: {'none' if NOPE else 'rope'}")
print0(f"compile mode: {DELTA_COMPILE_MODE}")
print0(f"KDA blocks include MLPs: {DELTA_MLP_ON_DELTA}")
print0(f"MLP hidden width: {MLP_HIDDEN}")
print0(
    f"DiffusionBlocks: {NUM_BLOCKS} blocks {BLOCK_LAYERS}; gamma={DBLOCK_GAMMA}; "
    f"sigma in [{DBLOCK_SIGMA_MIN}, {DBLOCK_SIGMA_MAX}], "
    f"lognormal({DBLOCK_P_MEAN}, {DBLOCK_P_STD}^2), sigma_data={DBLOCK_SIGMA_DATA}"
)
print0(
    "DiffusionBlocks sigma boundaries (ascending): "
    + ", ".join(f"{sigma:.4g}" for sigma in SIGMA_BOUNDARIES)
)
print0(
    "DiffusionBlocks training ranges by block (layer order, high noise first): "
    + ", ".join(
        f"{index}:[{low:.4g},{high:.4g}]"
        for index, (low, high) in enumerate(BLOCK_SIGMA_RANGES)
    )
)
print0(
    f"DiffusionBlocks inference: {DBLOCK_INFER_STEPS} Euler steps at sigmas "
    + ", ".join(f"{sigma:.4g}" for sigma in INFERENCE_SIGMAS)
    + f"; val noise seed {DBLOCK_VAL_SEED}"
)
print0(
    f"DiffusionBlocks chain train: {DBLOCK_CHAIN_TRAIN}; "
    f"clean propagation: {DBLOCK_CLEAN_PROP}; "
    f"readout gain: {DBLOCK_READOUT_GAIN}; "
    f"DiT fidelity: {DBLOCK_DIT_FIDELITY}; cond head: {DBLOCK_COND_HEAD}"
)
print0("="*100)

data_path = os.environ.get("DATA_PATH", "data/datasets/fineweb10B_gpt2")

val_tokens = int(os.environ.get("VAL_TOKENS", 64 * 524288))
batch_size = int(os.environ.get("GLOBAL_BATCH_TOKENS", 8 * 64 * 1024))
seq_len = int(os.environ.get("SEQ_LEN", 1024))
mbs = int(os.environ.get("MBS", 8))
if batch_size <= 0:
    raise ValueError(
        f"GLOBAL_BATCH_TOKENS must be positive, got {batch_size}"
    )
assert batch_size % (world_size * mbs * seq_len) == 0
assert val_tokens % (world_size * mbs * seq_len) == 0
local_microbatches_per_step = batch_size // (
    dist.get_world_size() * mbs * seq_len
)
print0(
    f"batch: global_tokens={batch_size}, sequence_length={seq_len}, "
    f"microbatch_sequences={mbs}, "
    f"local_microbatches_per_step={local_microbatches_per_step}"
)
val_glob = os.environ.get("VAL_GLOB", "fineweb_val_*.bin")
val_inputs, val_targets = next(distributed_data_generator(
    f"{data_path}/{val_glob}", val_tokens, seq_len=seq_len))
print0(f"validation panel: {data_path}/{val_glob}")

dataset_manifest = read_dataset_manifest(data_path)
dataset_manifest_path = Path(data_path) / "mix_manifest.json"
dataset_provenance = {
    "manifest_sha256": (
        hashlib.sha256(dataset_manifest_path.read_bytes()).hexdigest()
        if dataset_manifest_path.is_file()
        else None
    ),
    "payload_sha256": dataset_manifest.get("payload_sha256"),
    "source_manifest_sha256": dataset_manifest.get("source_manifest", {}).get(
        "sha256"
    ),
    "source_manifest_payload_sha256": dataset_manifest.get(
        "source_manifest_payload_sha256"
    ),
}
require_matching_vocab_size(dataset_manifest, VOCAB_SIZE)
byte_counter = ByteCounter(dataset_manifest, device=device)
if byte_counter.expected_lut_size() is not None:
    assert byte_counter.expected_lut_size() == VOCAB_SIZE
with torch.no_grad():
    val_byte_count_tensor = torch.tensor(
        byte_counter.count(val_targets), dtype=torch.int64, device=device
    )
    dist.all_reduce(val_byte_count_tensor, op=dist.ReduceOp.SUM)
    val_byte_count = float(val_byte_count_tensor)
    assert val_byte_count > 0

domain_val_tokens = int(os.environ.get("DOMAIN_VAL_TOKENS", "1048576"))
assert domain_val_tokens % (world_size * mbs * seq_len) == 0
domain_validation = {}
for domain in ("web", "code", "math", "knowledge"):
    pattern = f"{data_path}/domainval_{domain}_*.bin"
    if list(Path.cwd().glob(pattern)):
        domain_inputs, domain_targets = next(
            distributed_data_generator(
                pattern,
                domain_val_tokens,
                seq_len=seq_len,
            )
        )
        with torch.no_grad():
            domain_bytes_tensor = torch.tensor(
                byte_counter.count(domain_targets),
                dtype=torch.int64,
                device=device,
            )
            dist.all_reduce(domain_bytes_tensor, op=dist.ReduceOp.SUM)
            domain_bytes = float(domain_bytes_tensor)
        domain_validation[domain] = (
            domain_inputs,
            domain_targets,
            domain_bytes,
        )

model = DBlockGPT(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    model_dim=512,
).cuda()
# The two-stream layout is static per run: one flex_attention block mask and
# one rotary position vector serve every dense layer and every microbatch.
BLOCK_MASK = create_block_mask(
    dblock_mask_mod(seq_len),
    B=None,
    H=None,
    Q_LEN=2 * seq_len,
    KV_LEN=2 * seq_len,
    device=str(device),
)
POSITIONS = dblock_positions(seq_len, device)
# Compile every residual block independently so an opaque FLA recurrence
# cannot make Dynamo abandon the entire fixed-depth model. Compile the EDM
# head and cross-entropy tail separately; they dominate the non-block work
# and benefit heavily from fusion (the baseline's compute_loss pattern).
for block in model.blocks:
    block.compile(dynamic=False, mode=DELTA_COMPILE_MODE)
model.head_logits = torch.compile(
    model.head_logits, dynamic=False, mode=DELTA_COMPILE_MODE
)
model.loss_tail = torch.compile(
    model.loss_tail, dynamic=False, mode=DELTA_COMPILE_MODE
)
model.loss_tail_chain = torch.compile(
    model.loss_tail_chain, dynamic=False, mode=DELTA_COMPILE_MODE
)
print0(f"parameters: {sum(p.numel() for p in model.parameters()):,}", console=True)
print0(f"val window: {val_tokens:,} tokens = {val_byte_count:,.0f} bytes "
       f"({val_byte_count/val_tokens:.3f} bytes/token)", console=True)


def sampled_block_index(step: int, seed: int) -> int:
    """Uniform block choice as a pure function of the step, so resume replays
    the identical block schedule without touching the checkpointed RNG."""
    block_generator = torch.Generator()
    block_generator.manual_seed(seed * 1_000_003 + step)
    return int(torch.randint(NUM_BLOCKS, (), generator=block_generator))


num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for trial in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    train_steps = int(os.environ.get("ITERATIONS", 1000))
    stop_after_step = int(os.environ.get("STOP_AFTER_STEP", str(train_steps)))
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT")
    save_resume_state = os.environ.get("SAVE_RESUME_STATE", "0") == "1"
    if not 0 < stop_after_step <= train_steps:
        raise ValueError(
            f"STOP_AFTER_STEP must be in (0, {train_steps}], got {stop_after_step}"
        )
    if world_size > 1 and (
        resume_checkpoint or save_resume_state or stop_after_step < train_steps
    ):
        raise NotImplementedError(
            "exact resume is currently single-rank only because Muon state is "
            "rank-sharded; run multi-GPU training uninterrupted"
        )
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 20))
    domain_val_every = int(os.environ.get("DOMAIN_VAL_EVERY", "200"))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))
    if min(val_loss_every, domain_val_every, train_log_every) <= 0:
        raise ValueError("validation and training log intervals must be positive")
    if domain_val_every % val_loss_every:
        raise ValueError(
            "DOMAIN_VAL_EVERY must be a multiple of VAL_LOSS_EVERY because "
            "domain evaluation runs inside the validation section"
        )

    seed = int(os.environ.get("SEED", 1337))
    initialize_model(model, seed)
    model.zero_grad(set_to_none=True)

    # create the optimizer(s)
    embed_lr = float(os.environ.get("EMBED_LR", 0.7))
    proj_lr = float(os.environ.get("PROJ_LR", 0.004))
    delta_conv_lr = float(os.environ.get("KDA_CONV_LR", 0.004))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.015))
    muon_lr = float(os.environ.get("MUON_LR", 0.025))
    if min(embed_lr, proj_lr, delta_conv_lr, scalar_lr, muon_lr) < 0:
        raise ValueError("optimizer learning rates must be nonnegative")
    adam_weight_decay = float(os.environ.get("ADAM_WEIGHT_DECAY", 0.001))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.05))
    if adam_weight_decay < 0 or muon_weight_decay < 0:
        raise ValueError("optimizer weight decays must be nonnegative")

    # DiffusionBlocks trains one block per step, so optimizer state must be
    # partitioned the same way: shared modules (embedding, vocabulary head,
    # sigma embedder, global norms) step every iteration, while each block's
    # optimizers step only on that block's iterations. A single flat optimizer
    # would either crash on absent gradients or silently weight-decay and
    # momentum-decay blocks that did not train.
    shared_scalar_params = [
        model.norm1.gains,
        model.norm2.gains,
        model.time_embed.fc1.bias,
        model.time_embed.fc2.bias,
    ]
    shared_adam = AdamW(
        [
            dict(params=[model.embed.weight], lr=embed_lr),
            dict(params=[model.proj.weight], lr=proj_lr),
            dict(
                params=shared_scalar_params
                + [model.proj.bias]
                + (
                    [model.readout_gain]
                    if model.readout_gain is not None
                    else []
                ),
                lr=scalar_lr,
            ),
        ],
        betas=(0.8, 0.95), eps=1e-10,
        weight_decay=adam_weight_decay, fused=True,
    )
    shared_muon = Muon(
        [model.time_embed.fc1.weight, model.time_embed.fc2.weight]
        + ([model.head_ada.weight] if model.head_ada is not None else []),
        lr=muon_lr,
        weight_decay=muon_weight_decay,
        mu=float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85)),
    )

    block_optimizers: list[list[torch.optim.Optimizer]] = []
    block_param_ids: list[set[int]] = []
    for block_index, layer_indices in enumerate(BLOCK_LAYERS):
        block_params = [
            p
            for layer_idx in layer_indices
            for p in model.blocks[layer_idx].parameters()
        ]
        conv_params = [
            conv.weight
            for layer_idx in layer_indices
            if isinstance(model.blocks[layer_idx].attn, KimiDeltaAttention)
            for conv in (
                model.blocks[layer_idx].attn.q_conv1d,
                model.blocks[layer_idx].attn.k_conv1d,
                model.blocks[layer_idx].attn.v_conv1d,
            )
        ]
        conv_param_ids = {id(p) for p in conv_params}
        decay_params = [
            p
            for layer_idx in layer_indices
            if isinstance(model.blocks[layer_idx].attn, KimiDeltaAttention)
            for p in (
                model.blocks[layer_idx].attn.A_log,
                model.blocks[layer_idx].attn.dt_bias,
            )
        ]
        decay_param_ids = {id(p) for p in decay_params}
        scalar_params = [
            p for p in block_params
            if p.ndim < 2 and id(p) not in decay_param_ids
        ]
        matrix_params = [
            p for p in block_params
            if p.ndim >= 2 and id(p) not in conv_param_ids
        ]
        adam_groups = [dict(params=scalar_params, lr=scalar_lr)]
        if conv_params:
            adam_groups.append(dict(params=conv_params, lr=delta_conv_lr))
        if decay_params:
            adam_groups.append(
                dict(params=decay_params, lr=scalar_lr, weight_decay=0.0)
            )
        block_adam = AdamW(
            adam_groups,
            betas=(0.8, 0.95), eps=1e-10,
            weight_decay=adam_weight_decay, fused=True,
        )
        block_muon = Muon(
            matrix_params,
            lr=muon_lr,
            weight_decay=muon_weight_decay,
            mu=float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85)),
        )
        block_optimizers.append([block_adam, block_muon])
        block_param_ids.append({id(p) for p in block_params})

    shared_optimizers = [shared_adam, shared_muon]
    optimizers = shared_optimizers + [
        opt for pair in block_optimizers for opt in pair
    ]
    shared_param_ids = {
        id(p)
        for optimizer in shared_optimizers
        for group in optimizer.param_groups
        for p in group["params"]
    }
    owned_parameters = [
        parameter
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(owned_parameters) == len({id(p) for p in owned_parameters}), (
        "a parameter is owned by more than one optimizer group"
    )
    assert set(owned_parameters) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85)
    )
    muon_momentum_warmup_steps = int(
        os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500)
    )
    if not 0 <= muon_momentum_warmup_start <= muon_momentum < 1:
        raise ValueError(
            "Muon momentum must satisfy "
            "0 <= warmup_start <= momentum < 1"
        )
    if muon_momentum_warmup_steps < 0:
        raise ValueError("MUON_MOMENTUM_WARMUP_STEPS must be nonnegative")

    lr_schedule = os.environ.get("LR_SCHEDULE", "stable_linear")
    if lr_schedule not in {"stable_linear", "cosine"}:
        raise ValueError(
            f"LR_SCHEDULE must be 'stable_linear' or 'cosine', got {lr_schedule!r}"
        )
    default_warmup_fraction = "0.01" if lr_schedule == "cosine" else "0"
    warmup_fraction = float(
        os.environ.get("WARMUP_FRACTION", default_warmup_fraction)
    )
    if not 0 <= warmup_fraction < 1:
        raise ValueError(
            f"WARMUP_FRACTION must be in [0, 1), got {warmup_fraction}"
        )
    print0(
        f"LR schedule: {lr_schedule}; warmup fraction: {warmup_fraction}; "
        f"embed/proj/conv/scalar/Muon LRs: "
        f"{embed_lr}/{proj_lr}/{delta_conv_lr}/{scalar_lr}/{muon_lr}; "
        f"Adam weight decay: {adam_weight_decay}; "
        f"Muon weight decay: {muon_weight_decay}; "
        f"Muon momentum: {muon_momentum_warmup_start}"
        f"->{muon_momentum} over {muon_momentum_warmup_steps} steps",
        console=True,
    )

    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < warmup_fraction:
            eta = (step + 1) / max(1, round(train_steps * warmup_fraction))
        elif lr_schedule == "cosine":
            post_warmup = (progress - warmup_fraction) / (1 - warmup_fraction)
            eta = 0.5 * (1 + math.cos(math.pi * post_warmup))
        elif progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta
                if "mu" in group:
                    momentum_progress = (
                        min(step / muon_momentum_warmup_steps, 1.0)
                        if muon_momentum_warmup_steps
                        else 1.0
                    )
                    group["mu"] = (
                        muon_momentum_warmup_start
                        + momentum_progress
                        * (muon_momentum - muon_momentum_warmup_start)
                    )

    # The DiffusionBlocks recipe axes a checkpoint's chain semantics depend
    # on. Both resume and evaluation-only loads validate them: a mismatch
    # (including a checkpoint written before an axis existed) is a loud
    # error, never a silently different chain.
    expected_dblock_config = {
        "dblock_num_blocks": NUM_BLOCKS,
        "dblock_infer_steps": DBLOCK_INFER_STEPS,
        "dblock_block_layers": BLOCK_LAYERS,
        "dblock_gamma": DBLOCK_GAMMA,
        "dblock_sigma_min": DBLOCK_SIGMA_MIN,
        "dblock_sigma_max": DBLOCK_SIGMA_MAX,
        "dblock_p_mean": DBLOCK_P_MEAN,
        "dblock_p_std": DBLOCK_P_STD,
        "dblock_sigma_data": DBLOCK_SIGMA_DATA,
        "dblock_dit_fidelity": DBLOCK_DIT_FIDELITY,
        "dblock_cond_head": DBLOCK_COND_HEAD,
        "dblock_chain_train": DBLOCK_CHAIN_TRAIN,
        "dblock_clean_prop": DBLOCK_CLEAN_PROP,
        "dblock_readout_gain": DBLOCK_READOUT_GAIN,
    }

    def check_dblock_config(payload: dict, source: str) -> None:
        saved_model_config = payload.get("model_config", {})
        saved_dblock_config = {
            key: saved_model_config.get(key)
            for key in expected_dblock_config
        }
        if saved_dblock_config != expected_dblock_config:
            raise ValueError(
                f"{source} DiffusionBlocks config differs from the current "
                f"recipe: saved={saved_dblock_config}, "
                f"current={expected_dblock_config}"
            )

    start_step = 0
    if resume_checkpoint:
        resume_payload = torch.load(
            resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if "optimizer_states" not in resume_payload:
            raise ValueError(
                f"{resume_checkpoint} does not contain optimizer_states"
            )
        saved_seed = resume_payload.get("training_config", {}).get("seed")
        if saved_seed != seed:
            raise ValueError(
                f"resume seed differs from the current run: saved={saved_seed}, "
                f"current={seed}. The per-step diffusion-block schedule is a "
                "pure function of (seed, step), so resuming under a different "
                "seed would silently train different blocks than the "
                "uninterrupted run."
            )
        saved_batch_tokens = resume_payload.get("training_config", {}).get(
            "global_batch_tokens"
        )
        if saved_batch_tokens is not None and saved_batch_tokens != batch_size:
            raise ValueError(
                "resume global batch differs from the current run: "
                f"saved={saved_batch_tokens}, current={batch_size}. The data "
                "stream is indexed by step * global_batch_tokens, so this "
                "would not continue the same training run."
            )
        saved_val_seed = resume_payload.get("training_config", {}).get(
            "dblock_val_seed"
        )
        if saved_val_seed != DBLOCK_VAL_SEED:
            raise ValueError(
                "resume validation noise seed differs from the current run: "
                f"saved={saved_val_seed}, current={DBLOCK_VAL_SEED}. The val "
                "curve would silently stop being comparable across the "
                "resume boundary."
            )
        check_dblock_config(resume_payload, "resume")
        model.load_state_dict(resume_payload["model"], strict=True)
        optimizer_states = resume_payload["optimizer_states"]
        if len(optimizer_states) != len(optimizers):
            raise ValueError(
                f"resume has {len(optimizer_states)} optimizers, "
                f"current recipe has {len(optimizers)}"
            )
        for optimizer, state in zip(optimizers, optimizer_states, strict=True):
            optimizer.load_state_dict(state)
        start_step = int(resume_payload["completed_steps"])
        if start_step >= stop_after_step:
            raise ValueError(
                f"resume step {start_step} must be below STOP_AFTER_STEP "
                f"{stop_after_step}"
            )
        torch.set_rng_state(resume_payload["torch_rng_state"])
        torch.cuda.set_rng_state(
            resume_payload["cuda_rng_state"],
            device=device,
        )
        model.zero_grad(set_to_none=True)
        print0(
            f"resumed model, optimizers, and RNG from {resume_checkpoint} "
            f"at completed step {start_step}",
            console=True,
        )

    eval_checkpoint = os.environ.get("EVAL_CHECKPOINT", "")
    if eval_checkpoint:
        if resume_checkpoint:
            raise ValueError(
                "EVAL_CHECKPOINT and RESUME_CHECKPOINT both set; an evaluation "
                "run must not also continue training"
            )
        eval_payload = torch.load(
            eval_checkpoint, map_location="cpu", weights_only=False
        )
        check_dblock_config(eval_payload, "evaluation")
        model.load_state_dict(eval_payload["model"], strict=True)
        start_step = stop_after_step
        print0(
            f"evaluation-only: loaded {eval_checkpoint} "
            f"(completed_steps={eval_payload.get('completed_steps')}), "
            f"scoring one panel at step {stop_after_step} without training",
            console=True,
        )

    ########################################
    #        Training and Validation       #
    ########################################

    def run_chain_panel(
        panel_inputs: Tensor, panel_targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """(final CE sum, per-level CE sums) across a full panel."""
        panel_loss = torch.zeros((), device=device)
        level_losses = torch.zeros(len(INFERENCE_SIGMAS), device=device)
        assert len(panel_inputs) % mbs == 0
        for i in range(len(panel_inputs) // mbs):
            final_ce, per_level = chain_validation_loss(
                model,
                panel_inputs[i*mbs:(i+1)*mbs],
                panel_targets[i*mbs:(i+1)*mbs],
                BLOCK_MASK,
                POSITIONS,
                row_offset=i * mbs,
            )
            panel_loss += final_ce
            level_losses += torch.stack(per_level)
        return panel_loss, level_losses

    train_loader = distributed_data_generator(
        f"{data_path}/fineweb_train_*.bin",
        batch_size,
        seq_len=seq_len,
        start_step=start_step,
    )
    skip_initial_validation = (
        os.environ.get("SKIP_INITIAL_VALIDATION", "0") == "1"
    )
    if DBLOCK_DIAG and skip_initial_validation and start_step == stop_after_step:
        raise ValueError(
            "DBLOCK_DIAG with SKIP_INITIAL_VALIDATION would silently skip "
            "the diagnostic panel in an evaluation-only run"
        )
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    # start the clock
    training_time = 0
    last_val_step = start_step
    dist.barrier()
    t0 = time.perf_counter()
    for step in range(start_step, stop_after_step + 1):

        # --------------- VALIDATION SECTION -----------------
        if (
            step == stop_after_step or step % val_loss_every == 0
        ) and not (skip_initial_validation and step == start_step):
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            elapsed_steps = step - last_val_step
            step_avg = (
                time_since_last_val / elapsed_steps
                if elapsed_steps > 0
                else float("nan")
            )
            last_val_step = step
            training_time += time_since_last_val
            train_peak_vram_mib = torch.cuda.max_memory_allocated() / 2**20
            model.eval()
            with torch.no_grad():
                val_loss, val_levels = run_chain_panel(val_inputs, val_targets)
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_levels, op=dist.ReduceOp.SUM)
            val_loss /= val_tokens
            val_levels /= val_tokens
            val_bpb = (float(val_loss) / math.log(2.0)) * (val_tokens / val_byte_count)
            domain_bpb = {}
            if domain_validation and (
                step == stop_after_step or step % domain_val_every == 0
            ):
                for domain, (
                    domain_inputs,
                    domain_targets,
                    domain_bytes,
                ) in domain_validation.items():
                    with torch.no_grad():
                        domain_loss, _ = run_chain_panel(
                            domain_inputs, domain_targets
                        )
                    dist.all_reduce(domain_loss, op=dist.ReduceOp.SUM)
                    domain_loss /= domain_val_tokens
                    domain_bpb[domain] = (
                        float(domain_loss)
                        / math.log(2.0)
                        * (domain_val_tokens / domain_bytes)
                    )
            peak_vram_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
            peak_vram_reserved_mib = torch.cuda.max_memory_reserved() / 2**20
            process_peak_rss_mib = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            )
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} val_bpb:{val_bpb:.4f}"
                   + "".join(
                       f" val_level{level}_ce:{float(level_ce):.4f}"
                       for level, level_ce in enumerate(val_levels)
                   )
                   + f" train_time:{1000*training_time:.0f}ms step_avg:{1000*step_avg:.2f}ms"
                   + f" train_peak_vram_mib:{train_peak_vram_mib:.0f}"
                   + f" peak_vram_allocated_mib:{peak_vram_allocated_mib:.0f}"
                   + f" peak_vram_reserved_mib:{peak_vram_reserved_mib:.0f}"
                   + f" process_peak_rss_mib:{process_peak_rss_mib:.0f}"
                   + (
                       "".join(
                           f" rgain_l{level}:{value:.4f}"
                           for level, value in enumerate(
                               model.readout_gain.tolist()
                           )
                       )
                       if model.readout_gain is not None
                       else ""
                   )
                   + "".join(
                       f" domain_{domain}_bpb:{value:.4f}"
                       for domain, value in domain_bpb.items()
                   ),
                   console=True)
            if DBLOCK_DIAG and step == stop_after_step:
                def run_diag_panel(fn, *fn_args):
                    assert len(val_inputs) % mbs == 0
                    totals = None
                    with torch.no_grad():
                        for i in range(len(val_inputs) // mbs):
                            parts = fn(
                                model,
                                val_inputs[i*mbs:(i+1)*mbs],
                                val_targets[i*mbs:(i+1)*mbs],
                                BLOCK_MASK,
                                POSITIONS,
                                i * mbs,
                                *fn_args,
                            )
                            if not isinstance(parts, tuple):
                                parts = (parts,)
                            stacked = [torch.stack(p) for p in parts]
                            if totals is None:
                                totals = stacked
                            else:
                                totals = [
                                    t + s for t, s in zip(totals, stacked)
                                ]
                    for t in totals:
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    return totals

                # Diagnostics must never cost a completed run its
                # checkpoint: report any probe failure and fall through to
                # the normal save path.
                try:
                    (oracle_ce,) = run_diag_panel(oracle_level_losses)
                    print0(
                        "diag_oracle_ce:"
                        + "".join(
                            f" level{k}:{float(v)/val_tokens:.4f}"
                            for k, v in enumerate(oracle_ce)
                        ),
                        console=True,
                    )
                    fine_sigmas = inference_sigma_schedule(
                        DBLOCK_DIAG_FINE_STEPS,
                        sigma_min=DBLOCK_SIGMA_MIN,
                        sigma_max=DBLOCK_SIGMA_MAX,
                        p_mean=DBLOCK_P_MEAN,
                        p_std=DBLOCK_P_STD,
                    )
                    for estimate, diag_sigmas in (
                        ("expectation", list(INFERENCE_SIGMAS)),
                        ("renorm", list(INFERENCE_SIGMAS)),
                        ("sample", list(INFERENCE_SIGMAS)),
                        ("expectation", fine_sigmas),
                    ):
                        diag_ce, est_norm, est_cos = run_diag_panel(
                            diagnostic_chain_loss, diag_sigmas, estimate
                        )
                        print0(
                            f"diag_chain estimate:{estimate}"
                            + f" steps:{len(diag_sigmas)}"
                            + f" final_ce:{float(diag_ce[-1])/val_tokens:.4f}"
                            + "".join(
                                f" level{k}_ce:{float(v)/val_tokens:.4f}"
                                for k, v in enumerate(diag_ce)
                            )
                            + "".join(
                                f" level{k}_norm:{float(v)/val_tokens:.4f}"
                                for k, v in enumerate(est_norm)
                            )
                            + "".join(
                                f" level{k}_cos:{float(v)/val_tokens:.4f}"
                                for k, v in enumerate(est_cos)
                            ),
                            console=True,
                        )
                except Exception:
                    import traceback
                    print0(
                        "diagnostic panel failed:\n"
                        + traceback.format_exc(),
                        console=True,
                    )
            model.train()
            # Peak statistics are reset here so the next report separates the
            # training window's footprint (the DiffusionBlocks claim under
            # test) from the validation chain's own transient peaks.
            torch.cuda.reset_peak_memory_stats()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        if step == stop_after_step:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        assert len(inputs) % mbs == 0
        train_loss_sum = torch.zeros((), device=device)
        train_ce_sum = torch.zeros((), device=device)
        if CHAIN_TRAIN_ALL:
            # Fused cascade: one Euler rollout per microbatch trains every
            # block in level order. Block b's grad-forward logits are graded
            # against the targets and backpropagated immediately (so only one
            # block's graph is ever alive), then reused detached for the
            # Euler step that produces block b+1's input — the prefix is
            # never recomputed. Gradients stay blockwise: z carries no
            # history, so each CE touches only its block plus shared modules.
            level_ce_sums = torch.zeros(NUM_BLOCKS, device=device)
            with torch.no_grad():
                # Weights are constant within a step (one optimizer step at
                # the end), so one table serves every microbatch and level.
                table_bf16 = normalized_embedding_table(
                    model.embed.weight
                ).to(torch.bfloat16)
            for i in range(len(inputs) // mbs):
                microbatch_inputs = inputs[i*mbs:(i+1)*mbs]
                microbatch_targets = targets[i*mbs:(i+1)*mbs]
                with torch.no_grad():
                    z = (
                        torch.randn(
                            mbs, microbatch_inputs.size(1), model.model_dim,
                            dtype=torch.float32, device=device,
                        )
                        * INFERENCE_SIGMAS[0]
                    )
                clean = None
                for level, sigma in enumerate(INFERENCE_SIGMAS):
                    sigma_rows = torch.full(
                        (mbs,), sigma, dtype=torch.float32, device=device
                    )
                    last_level = level + 1 == NUM_BLOCKS
                    ce, denoised, clean = model.dblock_cascade_level(
                        microbatch_inputs,
                        microbatch_targets,
                        level,
                        z,
                        sigma_rows,
                        BLOCK_MASK,
                        POSITIONS,
                        table_bf16,
                        not last_level,
                        clean,
                    )
                    ce.backward()
                    level_ce_sums[level] += ce.detach()
                    if not last_level:
                        with torch.no_grad():
                            slope = (z - denoised) / sigma
                            z = z + (INFERENCE_SIGMAS[level + 1] - sigma) * slope
                    del denoised
            train_loss_sum = level_ce_sums.sum() / NUM_BLOCKS
            train_ce_sum = level_ce_sums[NUM_BLOCKS - 1].clone()
            for name, p in model.named_parameters():
                assert p.grad is not None, name
                if world_size > 1:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            set_hparams(step)
            for opt in optimizers:
                opt.step()
        else:
            block_index = sampled_block_index(step, seed)
            sigma_low, sigma_high = BLOCK_SIGMA_RANGES[block_index]
            active_param_ids = shared_param_ids | block_param_ids[block_index]
            for i in range(len(inputs) // mbs):
                microbatch_inputs = inputs[i*mbs:(i+1)*mbs]
                microbatch_targets = targets[i*mbs:(i+1)*mbs]
                if CHAIN_TRAIN_SINGLE:
                    chain_z = chain_training_input(
                        model,
                        microbatch_inputs,
                        block_index,
                        BLOCK_MASK,
                        POSITIONS,
                    )
                    sigma_rows = torch.full(
                        (mbs,),
                        INFERENCE_SIGMAS[block_index],
                        dtype=torch.float32,
                        device=device,
                    )
                    weighted_loss, raw_ce = model.dblock_chain_loss(
                        microbatch_inputs,
                        microbatch_targets,
                        block_index,
                        chain_z,
                        sigma_rows,
                        BLOCK_MASK,
                        POSITIONS,
                    )
                else:
                    sigma_rows = sample_block_sigmas(
                        sigma_low,
                        sigma_high,
                        mbs,
                        DBLOCK_P_MEAN,
                        DBLOCK_P_STD,
                        device,
                    )
                    weighted_loss, raw_ce = model.dblock_loss(
                        microbatch_inputs,
                        microbatch_targets,
                        block_index,
                        sigma_rows,
                        BLOCK_MASK,
                        POSITIONS,
                    )
                train_loss_sum += weighted_loss.detach()
                train_ce_sum += raw_ce
                weighted_loss.backward()
            for name, p in model.named_parameters():
                if id(p) not in active_param_ids:
                    assert p.grad is None, (
                        f"{name} received a gradient outside its block"
                    )
                    continue
                assert p.grad is not None, name
                if world_size > 1:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            # set optimization hyperparameters and take a step
            set_hparams(step)
            for opt in shared_optimizers + block_optimizers[block_index]:
                opt.step()
        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        if (step + 1) % train_log_every == 0:
            train_loss = float(train_loss_sum) / targets.numel()
            train_ce = float(train_ce_sum) / targets.numel()
            if CHAIN_TRAIN_ALL:
                # Numeric keys only: the ablation-parser extras grammar
                # rejects non-scalar tokens like ``block:all``.
                block_tag = " ".join(
                    f"ce_l{k}:{float(v)/targets.numel():.4f}"
                    for k, v in enumerate(level_ce_sums)
                )
            else:
                block_tag = f"block:{block_index}"
            print0(f"step:{step+1}/{train_steps} train_loss:{train_loss:.4f}"
                   + f" train_ce:{train_ce:.4f}"
                   + f" {block_tag}"
                   + f" train_time:{1000*approx_training_time:.0f}ms"
                   + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True)

    # Self-describing pretraining checkpoint.
    if dist.get_rank() == 0:
        suffix = f"_trial{trial}" if num_trials > 1 else ""
        ckpt_path = f"logs/{run_id}{suffix}_final_model.pt"
        model_payload = {
            "model": model.state_dict(),
            "model_config": dict(
                vocab_size=VOCAB_SIZE,
                num_layers=NUM_LAYERS,
                model_dim=512,
                mlp_hidden=MLP_HIDDEN,
                delta_num_heads=KDA_NUM_HEADS,
                delta_full_rank_gate=KDA_FULL_RANK_GATE,
                delta_attention_type="kda",
                delta_layer_indices=sorted(DELTA_LAYER_INDICES),
                delta_mlp_on_delta=DELTA_MLP_ON_DELTA,
                dense_attention_type="mha",
                dense_position_encoding="none" if NOPE else "rope",
                dblock_num_blocks=NUM_BLOCKS,
                dblock_block_layers=BLOCK_LAYERS,
                dblock_gamma=DBLOCK_GAMMA,
                dblock_sigma_min=DBLOCK_SIGMA_MIN,
                dblock_sigma_max=DBLOCK_SIGMA_MAX,
                dblock_p_mean=DBLOCK_P_MEAN,
                dblock_p_std=DBLOCK_P_STD,
                dblock_sigma_data=DBLOCK_SIGMA_DATA,
                dblock_dit_fidelity=DBLOCK_DIT_FIDELITY,
                dblock_cond_head=DBLOCK_COND_HEAD,
                dblock_chain_train=DBLOCK_CHAIN_TRAIN,
                dblock_clean_prop=DBLOCK_CLEAN_PROP,
                dblock_readout_gain=DBLOCK_READOUT_GAIN,
                dblock_infer_steps=DBLOCK_INFER_STEPS,
                tokenizer_provenance=dataset_manifest.get(
                    "tokenizer_provenance"
                ),
                pretraining_data_path=data_path,
            ),
            "dataset_provenance": dataset_provenance,
            "architecture": (
                f"nanogpt_mini_gpt2vocab_kda_dblock{NUM_BLOCKS}_"
                + "".join(
                    "k" if layer_idx in DELTA_LAYER_INDICES else "d"
                    for layer_idx in range(NUM_LAYERS)
                )
                + ("_nope" if NOPE else "")
                + ("_cleanprop" if DBLOCK_CLEAN_PROP else "")
                + ("_rgain" if DBLOCK_READOUT_GAIN else "")
                + (
                    "_fullblocks_v1"
                    if DELTA_MLP_ON_DELTA
                    else "_mixers_v1"
                )
            ),
            "training_config": {
                # The per-step diffusion-block schedule is a pure function of
                # (seed, step); resume validates it so a changed SEED cannot
                # silently retrain different blocks than the original run.
                "seed": seed,
                "delta_disable_recompute": DELTA_DISABLE_RECOMPUTE,
                "delta_state_v_first": DELTA_STATE_V_FIRST,
                "delta_blockwise_compile": DELTA_BLOCKWISE_COMPILE,
                "delta_compile_mode": DELTA_COMPILE_MODE,
                "delta_mlp_on_delta": DELTA_MLP_ON_DELTA,
                "kda_num_heads": KDA_NUM_HEADS,
                "kda_full_rank_gate": KDA_FULL_RANK_GATE,
                "dense_attention_type": "mha",
                "dense_position_encoding": "none" if NOPE else "rope",
                "dblock_val_seed": DBLOCK_VAL_SEED,
                "lr_schedule": lr_schedule,
                "warmup_fraction": warmup_fraction,
                "embed_lr": embed_lr,
                "proj_lr": proj_lr,
                "delta_conv_lr": delta_conv_lr,
                "scalar_lr": scalar_lr,
                "muon_lr": muon_lr,
                "adam_weight_decay": adam_weight_decay,
                "muon_weight_decay": muon_weight_decay,
                "muon_momentum": muon_momentum,
                "muon_momentum_warmup_start": muon_momentum_warmup_start,
                "muon_momentum_warmup_steps": muon_momentum_warmup_steps,
                "global_batch_tokens": batch_size,
                "microbatch_sequences": mbs,
                "local_microbatches_per_step": local_microbatches_per_step,
            },
            "train_seq_len": seq_len,
            "completed_steps": stop_after_step,
            "planned_train_steps": train_steps,
        }
        torch.save(model_payload, ckpt_path)
        print0(f"saved checkpoint: {ckpt_path}", console=True)
        if save_resume_state or stop_after_step < train_steps:
            resume_path = f"logs/{run_id}{suffix}_resume.pt"
            torch.save(
                {
                    **model_payload,
                    "optimizer_states": [
                        optimizer.state_dict() for optimizer in optimizers
                    ],
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(device),
                },
                resume_path,
            )
            print0(f"saved exact resume state: {resume_path}", console=True)

dist.destroy_process_group()

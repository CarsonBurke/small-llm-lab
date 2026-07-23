"""nanogpt_mini_cwd_tieddot_train.py

H33 combination fork: `nanogpt_mini_cwd_train.py` (June-19 record port,
1.2692 @ 2k full-val) with the untied proj head replaced by the tieddot
attached dot readout (won on the mini recipe: 1.2817 vs 1.2856). Deltas
vs the cwd port, all head-local: GPT class readout (scale zero-init +
per-token bias, rms-normed attached codebook, softcap kept), proj group
dropped from AdamW (readout scale/bias land in the other-aux Adam group,
lr 0.01 betas (0.8,0.997); embed still excluded from tail-EMA, readout
params INCLUDED), MASTER_PORT 29649. The record's "proj"-zero init loop
now only hits block projs; the uniform-start trick comes from the
zero-init scale instead. Original port docstring follows.

Mini-lineage port of the CURRENT track-3 record (modded-nanogpt
``records/track_3_optimization/results/20260619_cwd_rowfloor_tailema/
train_gpt_cwd_SOTA.py``, "SOAP-f1 + u/w-rowfloor + radial + CWD +
EMA-Nesterov + tail-EMA"), replacing nanogpt_mini_train.py's
tuned-baseline training system wholesale.  Per user direction: fully
aligned with the June 19 record except our parameter changes.

Kept from nanogpt_mini_train.py (our parameters):
  - scale 6L/512d, vocab 1024, head_dim 128, MLP 4x (record arch classes
    are IDENTICAL to the mini's — track 3 freezes architecture)
  - sp1024 data shards, fixed batch 524,288 tokens/step (same as record
    AND the challenge protocol: matched-step is matched-data)
  - challenge BPB val (byte-LUT), VAL_TOKENS env knob for the full-split
    window, ITERATIONS / VAL_LOSS_EVERY / RUN_ID / SEED / DATA_PATH knobs

Adopted from the record (the whole training system):
  - init: torch defaults + proj zero-init + depth-scaled mlp.fc
    (alpha 0.30) + CGI Rademacher antithetic norm-gain split
    (alpha 0.125, head-mean shrink 0.5; pair-from-layer scaled 6/12 -> 3)
  - AdamW(embed lr 0.3, proj lr 1/320, betas (0.8,0.99), eps 1e-10, wd 0)
  - bias-correction-free aux Adam: gains (0.8,0.99), other aux
    (0.8,0.997), attn.proj.bias (0.8,0.9965), all lr 0.01
  - Muon lr 0.0375 with SOAP-f1 on all hidden matrices (beta2 0.90,
    denom power 0.5), attn trust gate (proj-only, floor/fade scheduled),
    Newton-Schulz (12-iter, gram-normalized), radial split-scale
    (outward 0.5), per-row u/w floor (0.3825, rho 1.0), radius pin,
    cautious WD 0.025
  - EMA-Nesterov wrapper (stepsize 0.3 lr-scheduled, ema 0.99)
  - per-group PowerCool tails: lr = min(flat, c*(t_end-step)^1.2) with
    crossovers at the record's run-fractions (Adam groups 51.24%, Muon
    17.59% of t_end); c derived at runtime from the fraction
  - Muon mu warmup 0.85->0.95 / end-cooldown ->0.85, tail-EMA readout
    (lambda 0.6) — all step constants scaled by the 2000/2900 horizon
    ratio (fractions of the record's t_end 2900, rounded)

Val lines print ``val_bpb`` as the TAIL-EMA BLEND readout once the EMA
exists (the record's readout, what a submission would ship) and
``val_raw_bpb`` alongside; before the EMA starts they are identical.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import itertools
import math
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist

import sentencepiece as spm

import train_gpt  # byte-LUT builder for the challenge BPB metric


########################################
#       Record schedule constants      #
########################################
# Fractions of the record's t_end=2900; resolved against our train_steps.

FINAL_LR_POWER = 1.2
ADAM_DECAY_START_FRAC = 1486 / 2900   # all Adam groups cross flat->power here
MUON_DECAY_START_FRAC = 513.6 / 2900   # record's literal MUON_POWER_C crossover

MU = 0.95
MU_MIN = 0.85
MU_WARMUP_FRAC = 300 / 2900
MU_COOLDOWN_FRAC = 200 / 2900

MUON_LR = 0.0375
# SOAP=0 disables SOAP-f1 preconditioning (and with it the attn trust gate,
# which only ever gates SOAP output) while keeping every other record lever —
# the H31 decomposition knob. Default 1 = record behavior.
SOAP_ENABLED = os.environ.get("SOAP", "1") == "1"
TARGET_UW = 0.3825
SOAP_BETA2 = 0.90
SOAP_PRECONDITION_FREQUENCY = 1
SOAP_DENOM_POWER = 0.50
ATTN_EARLY_TRUST_FLOOR = 0.45
ATTN_EARLY_TRUST_CAP = 0.85
ATTN_TRUST_FLOOR_END_FRAC = 1375 / 2900
ATTN_TRUST_FLOOR_FADE_END_FRAC = 1625 / 2900
ATTN_TRUST_MIN_AGREE = 0.20
ATTN_TRUST_MIN_GRAD_ALIGN = 0.00
ATTN_TRUST_POWER = 1.00
RADIAL_OUTWARD_SCALE = 0.5
RADIAL_INWARD_SCALE = 1.0
ROWFLOOR = True
ROWFLOOR_RHO = 1.0
CWD = 0.025
HEAD_DIM = 128

EMA_NESTEROV_STEPSIZE = 0.3
EMA_NESTEROV_EMA = 0.99
EMA_NESTEROV_PREFILL_FRAC = 300 / 2900
EMA_NESTEROV_REST_GAP_FRAC = 950 / 2900   # rest_steps = train_steps - gap

TAILEMA_TAU_FRAC = 150 / 2900
TAILEMA_START_FRAC = 2400 / 2900
TAILEMA_LAMBDA = 0.6

DI_FC_ALPHA = 0.30            # depth-scaled mlp.fc init
CGI_ALPHA = 0.125             # Rademacher norm-gain split
CGI_PAIR_FROM_FRAC = 6 / 12   # record pairs from layer 6 of 12
CGI_HEAD_MEAN_SHRINK = 0.50


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert files, f"no shards match {filename_pattern}"
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    # cycle: the sp1024 slice holds ~1992 batches of 524288 tokens; upstream's
    # plain iter() would StopIteration-crash runs past that (e.g. 2000 steps).
    file_iter = itertools.cycle(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
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

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=HEAD_DIM):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    """Record-port trunk with the tieddot head (H33): the untied proj head is
    replaced by the tied attached dot readout that won on the mini recipe
    (job 270). Zero-init scale reproduces the record's zero-init-proj uniform
    start; softcap kept (tieddot parity)."""
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.readout_scale = nn.Parameter(torch.zeros(()))
        self.readout_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        z = self.norm2(x)
        codebook = F.rms_norm(self.embed.weight, (self.embed.embedding_dim,))
        logits = (z @ codebook.type_as(z).t()).float() * self.readout_scale + self.readout_bias
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################
# Everything below through EMA_Nesterov is the record's optimizer stack,
# verbatim except the schedule constants resolved from fractions above.

def gram_frobenius_norm_estimate(G: Tensor, keepdim: bool = False, eps: float = 1e-10) -> Tensor:
    X = G.float()
    gram = X.mT @ X if X.size(-2) > X.size(-1) else X @ X.mT
    return gram.norm(dim=(-2, -1), keepdim=keepdim).sqrt().clamp_min(eps)

def _ns_inner(X: Tensor) -> Tensor:
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / gram_frobenius_norm_estimate(X, keepdim=True, eps=1e-7).to(X.dtype)
    X = _ns_inner(X)
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def should_soap_param(name: str) -> bool:
    is_mlp_fc = name.endswith(".mlp.fc.weight")
    is_mlp_proj = name.endswith(".mlp.proj.weight")
    is_attn_proj = name.endswith(".attn.proj.weight")
    is_qkv = (
        name.endswith(".attn.q.weight")
        or name.endswith(".attn.k.weight")
        or name.endswith(".attn.v.weight")
    )
    return is_mlp_fc or is_mlp_proj or is_attn_proj or is_qkv  # SOAP on all hidden 2D matrices

def is_attn_proj_param(name: str) -> bool:
    return name.endswith(".attn.proj.weight")

def is_attn_param(name: str) -> bool:
    return (
        name.endswith(".attn.q.weight")
        or name.endswith(".attn.k.weight")
        or name.endswith(".attn.v.weight")
        or name.endswith(".attn.proj.weight")
    )

def tensor_cosine(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    a_f, b_f = a.float(), b.float()
    return (a_f * b_f).sum() / (a_f.norm() * b_f.norm()).clamp_min(eps)

def trust_gate(raw: Tensor, soap: Tensor, grad: Tensor, eps: float = 1e-8) -> Tensor:
    # SOAP is trusted when it still points with raw momentum and is at least as
    # gradient-aligned as raw momentum. This catches stale whitening bases.
    raw_grad = tensor_cosine(raw, grad, eps)
    soap_grad = tensor_cosine(soap, grad, eps)
    soap_raw = tensor_cosine(soap, raw, eps)

    agree_gate = ((soap_raw - ATTN_TRUST_MIN_AGREE) / (1 - ATTN_TRUST_MIN_AGREE)).clamp(0, 1)
    denom = (raw_grad - ATTN_TRUST_MIN_GRAD_ALIGN).clamp_min(eps)
    grad_gate = ((soap_grad - ATTN_TRUST_MIN_GRAD_ALIGN) / denom).clamp(0, 1)
    gate = (agree_gate * grad_gate).clamp(0, 1)
    if ATTN_TRUST_POWER != 1.0:
        gate = gate.pow(ATTN_TRUST_POWER)
    return gate

def early_trust_floor_for_step(step: int) -> float:
    if ATTN_TRUST_FLOOR_FADE_END_STEP <= ATTN_TRUST_FLOOR_END_STEP:
        return 0.0 if step >= ATTN_TRUST_FLOOR_FADE_END_STEP else ATTN_EARLY_TRUST_FLOOR
    if step < ATTN_TRUST_FLOOR_END_STEP:
        return ATTN_EARLY_TRUST_FLOOR
    if step >= ATTN_TRUST_FLOOR_FADE_END_STEP:
        return 0.0
    return ATTN_EARLY_TRUST_FLOOR * (
        ATTN_TRUST_FLOOR_FADE_END_STEP - step
    ) / (ATTN_TRUST_FLOOR_FADE_END_STEP - ATTN_TRUST_FLOOR_END_STEP)

def bounded_trust_gate(gate: Tensor, step: int) -> Tensor:
    floor = early_trust_floor_for_step(step)
    cap = ATTN_EARLY_TRUST_CAP if step < ATTN_TRUST_FLOOR_FADE_END_STEP else 1.0
    return gate.clamp(min=floor, max=cap)

def norm_preserving_blend(raw: Tensor, soap: Tensor, gate: Tensor, eps: float = 1e-8) -> Tensor:
    blended = raw + (soap - raw) * gate.to(raw.dtype)
    raw_norm = gram_frobenius_norm_estimate(raw, eps=eps)
    blended_norm = gram_frobenius_norm_estimate(blended, eps=eps)
    return (blended * (raw_norm / blended_norm).to(blended.dtype)).to(raw.dtype)

def scale_radial_update(update: Tensor, param: Tensor, eps: float = 1e-12) -> Tensor:
    update_f = update.float()
    param_f = param.float()
    denom = (param_f * param_f).sum().clamp_min(eps)
    coeff = (update_f * param_f).sum() / denom
    radial = coeff * param_f
    tangential = update_f - radial
    # p.add_(update, alpha=-lr), so actual movement is -update.
    # Outward movement means (-update) is aligned with p, i.e. coeff < 0.
    radial_scale = torch.where(
        coeff < 0,
        update_f.new_tensor(RADIAL_OUTWARD_SCALE),
        update_f.new_tensor(RADIAL_INWARD_SCALE),
    )
    return (tangential + radial_scale * radial).to(update.dtype)

def target_radius_after_update(param: Tensor, update: Tensor, lr: float, eps: float = 1e-8) -> Tensor:
    param_f = param.float()
    update_f = update.float()
    before_norm = param_f.norm().clamp_min(eps)
    # Use only the radial component's first-order radius change as the intended
    # radius change; the post-step rescale below removes finite tangent drift.
    radial_delta = -lr * (update_f * param_f).sum() / before_norm
    return (before_norm + radial_delta).clamp_min(eps)

def rescale_to_radius(param: Tensor, target_norm: Tensor, eps: float = 1e-8):
    after_norm = param.float().norm().clamp_min(eps)
    param.mul_((target_norm / after_norm).to(param.dtype))

def soap_eigenbasis(mat: Tensor) -> Tensor:
    try:
        _, q = torch.linalg.eigh(mat + 1e-30 * torch.eye(mat.size(0), device=mat.device))
    except RuntimeError:
        _, q = torch.linalg.eigh(mat.double() + 1e-30 * torch.eye(mat.size(0), device=mat.device))
        q = q.float()
    return torch.flip(q, [1])

def soap_basis_qr(row_gg, col_gg, q_row, q_col, exp_avg_sq):
    row_eig = torch.diag(q_row.T @ row_gg @ q_row)
    row_sort = torch.argsort(row_eig, descending=True)
    q_row = q_row[:, row_sort]
    exp_avg_sq = exp_avg_sq.index_select(0, row_sort)
    q_row, _ = torch.linalg.qr(row_gg @ q_row)

    col_eig = torch.diag(q_col.T @ col_gg @ q_col)
    col_sort = torch.argsort(col_eig, descending=True)
    q_col = q_col[:, col_sort]
    exp_avg_sq = exp_avg_sq.index_select(1, col_sort)
    q_col, _ = torch.linalg.qr(col_gg @ q_col)
    return q_row, q_col, exp_avg_sq

def soap_precondition_momentum(update, state, beta2=SOAP_BETA2, eps=1e-8):
    update_f = update.float()
    if state["q_row"] is None:
        return update
    q_row, q_col = state["q_row"], state["q_col"]
    projected = q_row.T @ update_f @ q_col
    state["exp_avg_sq"].mul_(beta2).add_(projected.square(), alpha=1 - beta2)
    denom = state["exp_avg_sq"].clamp_min(eps * eps).pow(SOAP_DENOM_POWER)
    precond = q_row @ (projected / denom) @ q_col.T
    precond.mul_(gram_frobenius_norm_estimate(update_f, eps=eps) / gram_frobenius_norm_estimate(precond, eps=eps))
    return precond.to(update.dtype)

def soap_update_preconditioner(grad, state, shampoo_beta=SOAP_BETA2, precondition_frequency=SOAP_PRECONDITION_FREQUENCY):
    grad_f = grad.float()
    state["row_gg"].lerp_(grad_f @ grad_f.T, 1 - shampoo_beta)
    state["col_gg"].lerp_(grad_f.T @ grad_f, 1 - shampoo_beta)
    if state["q_row"] is None:
        state["q_row"] = soap_eigenbasis(state["row_gg"])
        state["q_col"] = soap_eigenbasis(state["col_gg"])
    elif state["soap_step"] > 0 and state["soap_step"] % precondition_frequency == 0:
        state["q_row"], state["q_col"], state["exp_avg_sq"] = soap_basis_qr(
            state["row_gg"], state["col_gg"], state["q_row"], state["q_col"], state["exp_avg_sq"]
        )
    state["soap_step"] += 1

def muon_update(update):
    # Newton-Schulz orthogonalization + aspect-ratio scale.
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, named_params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(named_params, list) and len(named_params) >= 1
        self.soap_params = {p for n, p in named_params if SOAP_ENABLED and should_soap_param(n)}
        self.attn_soap_params = {p for n, p in named_params if should_soap_param(n) and is_attn_param(n)}
        self.attn_proj_soap_params = {p for n, p in named_params if should_soap_param(n) and is_attn_proj_param(n)}
        self.step_count = 0
        params = sorted([p for _, p in named_params], key=lambda x: x.size(), reverse=True)
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
                        if p in self.soap_params:
                            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                            state["row_gg"] = torch.zeros(p.size(0), p.size(0), dtype=torch.float32, device=p.device)
                            state["col_gg"] = torch.zeros(p.size(1), p.size(1), dtype=torch.float32, device=p.device)
                            state["q_row"] = None
                            state["q_col"] = None
                            state["soap_step"] = 0
                    grad = p.grad
                    state["momentum"].lerp_(grad, 1 - group["mu"])
                    momentum_update = grad.lerp(state["momentum"], group["mu"])
                    is_attn_soap = p in self.attn_soap_params
                    use_soap = p in self.soap_params
                    if use_soap:
                        if is_attn_soap:
                            soap_update = soap_precondition_momentum(momentum_update, state)
                            if p in self.attn_proj_soap_params:
                                gate = bounded_trust_gate(
                                    trust_gate(momentum_update, soap_update, grad),
                                    self.step_count
                                )
                            else:
                                gate = torch.ones((), dtype=torch.float32, device=p.device)
                            momentum_update = norm_preserving_blend(momentum_update, soap_update, gate)
                        else:
                            momentum_update = soap_precondition_momentum(momentum_update, state)
                    update = muon_update(momentum_update)
                    update = scale_radial_update(update, p)
                    # u/w-floor.
                    p_fro = p.float().norm().clamp_min(1e-8)
                    u_fro = update.float().norm().clamp_min(1e-8)
                    cur_uw = u_fro / p_fro
                    target_uw = TARGET_UW
                    if ROWFLOOR and p.ndim == 2:
                        # RowFloor: boost each under-updated OUTPUT ROW to its target
                        # update/weight ratio. Per-row SHAPE change; magnitude is
                        # re-pinned below, so only the shape survives the radius pin.
                        r_row = p.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
                        s_row = update.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
                        f_row = torch.clamp(target_uw * r_row / s_row, min=1.0).pow(ROWFLOOR_RHO)
                        update = (update.float() * f_row).to(update.dtype)
                    else:
                        scale = torch.where(cur_uw < target_uw, target_uw * p_fro / u_fro, torch.ones_like(p_fro))
                        update = update * scale.to(update.dtype)
                    target_radius = target_radius_after_update(p, update, group["lr"])
                    # WD set to 0 — u/w target replaces wd's role.
                    if CWD > 0.0 and p.ndim == 2:
                        # Cautious Weight Decay: mask the coords where -lr*update already
                        # shrinks |p| (update*p>0); decay only those.
                        cwd_mask = (update.float() * p.float() > 0).to(p.dtype)
                    p.add_(update, alpha=-group["lr"])
                    rescale_to_radius(p, target_radius)
                    if CWD > 0.0 and p.ndim == 2:
                        # apply AFTER the radius pin (POST) so the per-coord shape change survives.
                        p.mul_(1.0 - (group["lr"] * CWD) * cwd_mask)
                    if use_soap:
                        soap_update_preconditioner(grad, state)
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self.step_count += 1


class Adam(torch.optim.Optimizer):
    """Bias-correction-free Adam for the aux (<2D) param groups (record's
    CenterShrinkAdam at rho=1.0; intentionally NOT torch AdamW)."""
    def __init__(self, params, lr=0.01, betas=(0.8, 0.99), eps=1e-10):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.lerp_(grad.float().square(), 1 - beta2)
                adam_dir = exp_avg.float() / (exp_avg_sq.sqrt() + eps)
                p.add_(adam_dir.to(p.dtype), alpha=-group["lr"])


class EMA_Nesterov(torch.optim.Optimizer):
    def __init__(self, params, inner_optimizer, lookahead_stepsize=0, use_scheduled_lookahead_stepsize=True, lookahead_ema=0.9, prefill_steps=0, rest_steps=0):
        if lookahead_stepsize < 0.0:
            raise ValueError("Invalid momentum value: {}".format(lookahead_stepsize))

        super(EMA_Nesterov, self).__init__(params, {})
        self.inner_optimizer = inner_optimizer
        self.use_scheduled_lookahead_stepsize = use_scheduled_lookahead_stepsize
        self.lookahead_stepsize = lookahead_stepsize
        self.lookahead_ema = lookahead_ema
        self.prefill_steps = prefill_steps
        self.rest_steps = rest_steps
        self.it = 0
        self.lookahead_status = False
        self.current_lookahead_stepsize = 0
        self.initialize_buffers()

    def __setstate__(self, state):
        super(EMA_Nesterov, self).__setstate__(state)

    @torch.no_grad()
    def initialize_buffers(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'prev_params' not in param_state:
                    param_state['prev_params'] = (p.clone(), self.it)

                if 'lookahead_buffer' not in param_state:
                    param_state['lookahead_buffer'] = (torch.zeros_like(p), -1)

    def get_lr_lambda(self):
        if isinstance(self.inner_optimizer, list):
            lr_lambda = self.inner_optimizer[0].param_groups[0]["lr"] / self.inner_optimizer[0].param_groups[0]["initial_lr"]
        else:
            lr_lambda = self.inner_optimizer.param_groups[0]["lr"] / self.inner_optimizer.param_groups[0]["initial_lr"]
        return lr_lambda

    @torch.no_grad()
    def lookahead_step(self):
        """Performs nesterov's lookahead."""
        if self.use_scheduled_lookahead_stepsize:
            lookahead_stepsize = self.lookahead_stepsize * self.get_lr_lambda()
        else:
            lookahead_stepsize = self.lookahead_stepsize
        self.current_lookahead_stepsize = lookahead_stepsize

        for group in self.param_groups:
            for p in group['params']:

                param_state = self.state[p]

                lookahead = param_state['lookahead_buffer'][0]

                p.add_(lookahead, alpha=lookahead_stepsize)

    @torch.no_grad()
    def accum_lookahead(self):
        """Update nesterov's lookahead direction."""
        lookahead_ema = self.lookahead_ema
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                look = p.add(param_state['prev_params'][0], alpha=-1)

                # update lookahead buffer
                buf = param_state['lookahead_buffer'][0]

                param_state['lookahead_buffer'] = (buf.lerp_(look, 1 - lookahead_ema), self.it) # m^{t+1} = beta * m^t + (1-beta) * look

                # update prev_params buffer
                param_state['prev_params'] = (param_state['prev_params'][0].copy_(p), self.it)

    @torch.no_grad()
    def nesterov_step(self):
        if self.it + 1 > self.prefill_steps and self.it < self.rest_steps and not self.lookahead_status:
            self.lookahead_step()
        else:
            self.current_lookahead_stepsize = 0
        self.lookahead_status = True

    @torch.no_grad()
    def step(self):
        if not self.lookahead_status:
            raise ValueError("optimizer.nesterov_step() should be invoked before model forward pass.")
        if isinstance(self.inner_optimizer, list):
            for opt in self.inner_optimizer:
                opt.step()
        else:
            self.inner_optimizer.step()

        self.accum_lookahead()

        self.lookahead_status = False
        self.it += 1


########################################
#                Setup                 #
########################################

# Single-process shim: default the torchrun env vars so `python3` launches work.
if "RANK" not in os.environ:
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29649")

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
SEED = int(os.environ.get("SEED", 1337))
torch.manual_seed(SEED)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

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
print0("="*100)

data_path = os.environ.get("DATA_PATH", "data/datasets/fineweb_onepass_sp1024")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "data/tokenizers/fineweb_1024_bpe.model")

val_tokens = int(os.environ.get("VAL_TOKENS", 20 * 524288))
batch_size = 8 * 64 * 1024
mbs = 64
val_inputs, val_targets = next(distributed_data_generator(f"{data_path}/fineweb_val_*.bin", val_tokens))

# Challenge BPB metric: fixed byte count of the val window via the same
# sentencepiece byte-LUT accounting used by train_gpt.py's eval.
sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = train_gpt.build_sentencepiece_luts(
    sp, vocab_size=1024, device=device
)
with torch.no_grad():
    _prev = val_inputs.reshape(-1).to(torch.int64)
    _tgt = val_targets.reshape(-1)
    _bytes = base_bytes_lut[_tgt].to(torch.int64)
    _bytes += (has_leading_space_lut[_tgt] & ~is_boundary_token_lut[_prev]).to(torch.int64)
    val_byte_count = float(_bytes.sum())
    assert val_byte_count > 0

model = GPT(vocab_size=1024, num_layers=6, model_dim=512).cuda()
model.compile(dynamic=False)
print0(f"parameters: {sum(p.numel() for p in model.parameters()):,}", console=True)


########################################
#       Init & Optim Hyperparams       #
########################################

train_steps = int(os.environ.get("ITERATIONS", 2000))
val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 200))
train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))

# Resolve the record's step constants against our horizon.
FINAL_SCHEDULE_STEPS = train_steps
ADAM_DECAY_START = round(ADAM_DECAY_START_FRAC * FINAL_SCHEDULE_STEPS)
MUON_DECAY_START = round(MUON_DECAY_START_FRAC * FINAL_SCHEDULE_STEPS)
MU_WARMUP_STEPS = round(MU_WARMUP_FRAC * train_steps)
MU_COOLDOWN_STEPS = round(MU_COOLDOWN_FRAC * train_steps)
ATTN_TRUST_FLOOR_END_STEP = round(ATTN_TRUST_FLOOR_END_FRAC * train_steps)
ATTN_TRUST_FLOOR_FADE_END_STEP = round(ATTN_TRUST_FLOOR_FADE_END_FRAC * train_steps)
EMA_NESTEROV_PREFILL = round(EMA_NESTEROV_PREFILL_FRAC * train_steps)
EMA_NESTEROV_REST = train_steps - round(EMA_NESTEROV_REST_GAP_FRAC * train_steps)
TAILEMA_TAU = max(1.0, round(TAILEMA_TAU_FRAC * train_steps))
TAILEMA_START = round(TAILEMA_START_FRAC * train_steps)
TAILEMA_END = train_steps

print0(f"record-port schedule: t_end={FINAL_SCHEDULE_STEPS} adam_decay_start={ADAM_DECAY_START} "
       f"muon_decay_start={MUON_DECAY_START} mu_warmup={MU_WARMUP_STEPS} mu_cooldown={MU_COOLDOWN_STEPS} "
       f"trust_floor_end={ATTN_TRUST_FLOOR_END_STEP} trust_fade_end={ATTN_TRUST_FLOOR_FADE_END_STEP} "
       f"nesterov_prefill={EMA_NESTEROV_PREFILL} nesterov_rest={EMA_NESTEROV_REST} "
       f"tailema_tau={TAILEMA_TAU} tailema_start={TAILEMA_START} tailema_end={TAILEMA_END}", console=True)

# initialize model parameters: record scheme = torch defaults + the following.
for name, p in model.named_parameters():
    if "proj" in name:
        p.data.zero_()

# depth-scaled mlp.fc init
_NUM_BLOCKS = len(model.blocks)
with torch.no_grad():
    for l_idx, block in enumerate(model.blocks):
        ramp = l_idx / (_NUM_BLOCKS - 1) if _NUM_BLOCKS > 1 else 0.0
        s_l = 1.0 - DI_FC_ALPHA * ramp
        block.mlp.fc.weight.data.mul_(s_l)

# CGI Rademacher channel-gain split (pair-from-layer scaled by depth fraction)
_CGI_PAIR_FROM_LAYER = round(CGI_PAIR_FROM_FRAC * _NUM_BLOCKS)

def headmean_antithetic_pair(shape, *, device, layer_idx: int, seed: int, head_dim: int = HEAD_DIM):
    width = shape[-1]
    if width % head_dim != 0:
        s0 = (torch.randint(0, 2, shape, device=device, dtype=torch.float32) * 2 - 1)
        return s0, -s0
    gen = torch.Generator(device=device)
    gen.manual_seed(0xA9170000 + seed * 1009 + layer_idx * 9176)
    heads0 = []
    heads1 = []
    for _ in range(width // head_dim):
        h0 = (torch.randint(0, 2, (head_dim,), device=device, generator=gen, dtype=torch.int64) * 2 - 1).float()
        plus0 = int((h0 > 0).sum().item())
        plus0 = int(round(head_dim / 2 + CGI_HEAD_MEAN_SHRINK * (plus0 - head_dim / 2)))
        plus0 = max(0, min(head_dim, plus0))
        h0 = torch.cat([
            torch.ones(plus0, device=device, dtype=torch.float32),
            -torch.ones(head_dim - plus0, device=device, dtype=torch.float32),
        ])
        h0 = h0[torch.randperm(head_dim, device=device, generator=gen)]
        plus1 = head_dim - plus0
        h1 = torch.cat([
            torch.ones(plus1, device=device, dtype=torch.float32),
            -torch.ones(head_dim - plus1, device=device, dtype=torch.float32),
        ])
        h1 = h1[torch.randperm(head_dim, device=device, generator=gen)]
        heads0.append(h0)
        heads1.append(h1)
    return torch.cat(heads0).reshape(shape), torch.cat(heads1).reshape(shape)

with torch.no_grad():
    pair_next = None
    for l_idx, block in enumerate(model.blocks):
        if l_idx >= _CGI_PAIR_FROM_LAYER and (l_idx - _CGI_PAIR_FROM_LAYER) % 2 == 1:
            s = pair_next
            pair_next = None
        elif l_idx >= _CGI_PAIR_FROM_LAYER:
            s, pair_next = headmean_antithetic_pair(
                block.norm1.gains.shape,
                device=block.norm1.gains.device,
                layer_idx=l_idx,
                seed=SEED,
            )
        else:
            s = (torch.randint(0, 2, block.norm1.gains.shape,
                               device=block.norm1.gains.device, dtype=torch.float32) * 2 - 1)
        block.norm1.gains.data.copy_((1.0 - CGI_ALPHA * s).to(block.norm1.gains.dtype))
        block.norm2.gains.data.copy_((1.0 + CGI_ALPHA * s).to(block.norm2.gains.dtype))

# create the optimizer(s) — record groups and LRs
# tieddot head: no untied proj -> the 1/320 proj group is dropped; the
# readout scale/bias (ndim<2) fall into the other-aux Adam group below.
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=float(os.environ.get("EMBED_LR", 0.3)))],
                   betas=(0.8, 0.99), eps=1e-10, weight_decay=0, fused=True)
gain_aux_params = [p for n, p in model.named_parameters() if p.ndim < 2 and n.endswith(".gains")]
attn_proj_bias_params = [p for n, p in model.named_parameters() if n.endswith(".attn.proj.bias")]
other_aux_params = [p for n, p in model.named_parameters()
                    if p.ndim < 2 and not n.endswith(".gains")
                    and not n.endswith(".attn.proj.bias")]
optimizer3 = Adam([
        dict(params=gain_aux_params, lr=0.01, betas=(0.8, 0.99)),
        dict(params=other_aux_params, lr=0.01, betas=(0.8, 0.997)),
        dict(params=attn_proj_bias_params, lr=0.01, betas=(0.8, 0.9965)),
    ],
    lr=0.01, betas=(0.8, 0.99), eps=1e-10)
optimizer2 = Muon([(n, p) for n, p in model.blocks.named_parameters() if p.ndim >= 2],
                  lr=MUON_LR, mu=MU)
optimizers = [optimizer1, optimizer2, optimizer3]
optimizers = [EMA_Nesterov(
        [p for p in model.parameters()],
        optimizers,
        lookahead_stepsize=EMA_NESTEROV_STEPSIZE,
        use_scheduled_lookahead_stepsize=True,
        lookahead_ema=EMA_NESTEROV_EMA,
        prefill_steps=EMA_NESTEROV_PREFILL,
        rest_steps=EMA_NESTEROV_REST,
    )]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers[0].inner_optimizer:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# Per-group PowerCool coefficients from the crossover fractions:
# lr = min(flat, c*(t_end-step)^1.2) with c = lr / (t_end-start)^1.2.
_adam_remaining = max(FINAL_SCHEDULE_STEPS - ADAM_DECAY_START, 1)
_muon_remaining = max(FINAL_SCHEDULE_STEPS - MUON_DECAY_START, 1)
for group in optimizer1.param_groups:
    group["power_c"] = group["initial_lr"] / _adam_remaining ** FINAL_LR_POWER
for group in optimizer3.param_groups:
    group["power_c"] = group["initial_lr"] / _adam_remaining ** FINAL_LR_POWER
for group in optimizer2.param_groups:
    group["power_c"] = group["initial_lr"] / _muon_remaining ** FINAL_LR_POWER

# learning rate schedule: stable then power decay (per-group)
def _lr(step, initial_lr, power_c, power=1.0):
    t_end = FINAL_SCHEDULE_STEPS
    flat_lr = initial_lr
    downward_lr = power_c * max(0.0, t_end - step) ** power
    return min(flat_lr, downward_lr)

def _muon_mu_at_step(step, train_steps):
    cd_start = train_steps - MU_COOLDOWN_STEPS
    if step < MU_WARMUP_STEPS:
        frac = step / max(MU_WARMUP_STEPS, 1)
        return MU_MIN + frac * (MU - MU_MIN)
    elif step > cd_start:
        frac = (step - cd_start) / max(MU_COOLDOWN_STEPS, 1)
        return MU - frac * (MU - MU_MIN)
    else:
        return MU

def set_hparams(step):
    progress = step / FINAL_SCHEDULE_STEPS
    assert 0 <= progress
    mu = _muon_mu_at_step(step, train_steps)
    for opt in optimizers[0].inner_optimizer:
        for group in opt.param_groups:
            group["lr"] = _lr(step, group["initial_lr"], group["power_c"], FINAL_LR_POWER)
    for group in optimizer2.param_groups:
        group["mu"] = mu


########################################
#        Training and Validation       #
########################################

train_loader = distributed_data_generator(f"{data_path}/fineweb_train_*.bin", batch_size)
for p in model.parameters():
    dist.broadcast(p.detach(), 0)
# start the clock
training_time = 0
last_val_step = 0
_tailema = None   # tail-EMA buffer: fp32 per non-embed param; lazy-init at TAILEMA_START
dist.barrier()
t0 = time.perf_counter()
for step in range(train_steps + 1):

    # --------------- VALIDATION SECTION -----------------
    if step == train_steps or step % val_loss_every == 0:
        # stop the clock
        dist.barrier()
        time_since_last_val = time.perf_counter() - t0
        step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
        last_val_step = step
        training_time += time_since_last_val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        raw_bpb = (float(val_loss) / math.log(2.0)) * (val_tokens / val_byte_count)
        # Tail-EMA readout: eval the partial blend (1-LAMBDA)*w + LAMBDA*ema.
        blend_bpb = raw_bpb
        if _tailema is not None:
            with torch.no_grad():
                stash = [p.detach().clone() for p in model.parameters()]
                for j, p in enumerate(model.parameters()):
                    if _tailema[j] is not None:
                        p.copy_(((1.0 - TAILEMA_LAMBDA) * p.float() + TAILEMA_LAMBDA * _tailema[j]).to(p.dtype))
                val_ema_loss = 0
                for i in range(len(val_inputs) // mbs):
                    val_ema_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
                dist.all_reduce(val_ema_loss, op=dist.ReduceOp.SUM)
                val_ema_loss /= val_tokens
                blend_bpb = (float(val_ema_loss) / math.log(2.0)) * (val_tokens / val_byte_count)
                for p, s in zip(model.parameters(), stash):
                    p.copy_(s)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} val_bpb:{blend_bpb:.4f}"
               + f" val_raw_bpb:{raw_bpb:.4f}"
               + f" readout_scale:{float(model.readout_scale):.4f}"
               + f" train_time:{1000*training_time:.0f}ms step_avg:{1000*step_avg:.2f}ms", console=True)
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if step == train_steps:
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    # accumulate across microbatches in case we are running with fewer than 8 gpus
    assert len(inputs) % mbs == 0
    optimizers[0].nesterov_step()
    train_loss_sum = torch.zeros((), device=device)
    for i in range(len(inputs) // mbs):
        loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
        # NaN guard: catch divergence the step it happens.
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"non-finite train loss at step {step} mb {i}: {loss.item()}")
        train_loss_sum += loss.detach()
        loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    # set optimization hyperparameters and take a step
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    # Tail-EMA buffer maintenance: ema += (w - ema)/TAU over [START, END), embed excluded.
    if TAILEMA_TAU > 0 and TAILEMA_START <= (step + 1) < TAILEMA_END:
        if _tailema is None:
            _tailema = [None if p is model.embed.weight else p.detach().float().clone()
                        for p in model.parameters()]
        else:
            for j, p in enumerate(model.parameters()):
                if _tailema[j] is not None:
                    _tailema[j].add_(p.detach().float() - _tailema[j], alpha=1.0 / TAILEMA_TAU)
    approx_training_time = training_time + (time.perf_counter() - t0)
    if (step + 1) % train_log_every == 0:
        train_loss = float(train_loss_sum) / targets.numel()
        print0(f"step:{step+1}/{train_steps} train_loss:{train_loss:.4f}"
               + f" train_time:{1000*approx_training_time:.0f}ms"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True)

dist.destroy_process_group()

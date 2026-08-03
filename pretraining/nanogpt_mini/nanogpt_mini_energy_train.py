"""pretraining/nanogpt_mini/nanogpt_mini_energy_train.py

The no-sigreg energy readout (the current best of the energy_readout family,
job 249) ported onto the nanogpt-mini baseline (``pretraining/nanogpt_mini/nanogpt_mini_train.py``,
itself a minimal fork of modded-nanogpt's fixed-arch optimization-track
baseline).  Single-factor experiment vs the baseline: the trunk (6L/512d
blocks, Muon, AdamW groups, schedule, data, val protocol) is untouched; the
input embedding path and the output head are replaced by the family's latent
machinery:

    token_latent = latent_projector(rms_norm(embed(ids)))      # trunk input
    z_hat        = prediction_projector(norm2(trunk(x)))       # prediction
    codebook c_k = latent_projector.inference(rms_norm(embed.weight))
    logit_k      = b_k - 1/2 * sum_d s_d (z_hat_d - c_{k,d})^2 # per-dim scale

with ``TokenProjector`` = Linear(512->2048) -> FP32BatchNorm1d -> GELU ->
Linear(2048->512), exactly the family's classes (imported, not copied).  The
main path runs the projector in train mode (batch statistics; this is also
what keeps the running stats alive), the codebook uses running statistics and
stays attached — gradients flow to embed and the projector.  Head form is
switchable via ENERGY_HEAD_FORM=distance|dot (family default: distance).

Intricacy resolutions vs the baseline fork (user-flagged checklist):
  (a) logit softcap: REMOVED with the head — the family head is capless
      (distance logits are one-sided; tanh-style caps saturate the target's
      logit exactly when its code is far).
  (b) untied embed/proj: ``proj`` (0.53M) is deleted; the codebook derives
      from ``embed`` through the projector — tied by construction.  Net
      params ~23.6M (+2x2.10M projectors, +1.5k head scalars, -proj).
  (c) init: upstream std-1 embed init kept — rms_norm makes embed scale
      moot.  Projector linears/BN re-init inside the seeded section via
      reset_parameters() (their family init).  Shared trunk params consume
      the RNG stream in the baseline's exact order (projectors register
      after norm2; upstream ``proj`` consumed no RNG — it was zero-init),
      so the trunk starts bit-identical to the baseline run at equal SEED.
  (d) precision: projectors and the energy head run under a local
      bf16 autocast — CastedLinear matmuls in bf16, BatchNorm pinned fp32
      by FP32BatchNorm1d — reproducing the family's global-autocast
      numerics; distance accumulation and CE are explicit fp32; the trunk
      keeps upstream's manual bf16 weight-cast style, no autocast.
  (e) optimizer routing: projector 2D weights join the Muon group (family
      routed projector matrices to Muon); every new ndim<2 param (BN
      affine, linear biases, s_d, b_k) lands in the existing AdamW
      scalar group (lr 0.015) automatically; embed keeps upstream lr 0.7
      (registered risk: the family ran its tied table at 0.05 — if the
      codebook thrashes, this is the first knob); the dead proj group
      (lr 0.004) is dropped.  The parity audit of the tied 1k run (H16)
      identified this regime anchoring as the top suspect (the per-dim
      scales never specialized: max/mean ~1.24 vs the family's ~1.44 with
      ~2.1x max/min), so the head/codebook regime is now switchable:
      EMBED_LR (0.7), EMBED_WD (0.001), EMBED_INIT_STD (1.0),
      HEAD_SCALAR_LR (0.015), HEAD_WD (0.001) — defaults reproduce the
      H16 run bit-for-bit; the family regime is EMBED_LR=0.05 EMBED_WD=0
      EMBED_INIT_STD=0.005 HEAD_SCALAR_LR=0.04 HEAD_WD=0.
  (f) loss reduction: sum-CE kept (upstream's unnormalized-gradient
      convention; Muon/Adam are scale-invariant to it).

Known semantic deviations from the family head (reviewed, intentional):
  - BN statistics source/ordering: the parity audit found the earlier
    caveat here was wrong — the family also updates BN from the main
    input path before the codebook's inference() reads the running
    stats, so the port is faithful on this point (no deviation).
  - BN running stats are per-rank (no SyncBatchNorm), inherited from the
    family.  Irrelevant at world_size=1; revisit before any multi-GPU
    validation run.

s_d init = -0.5*ln(512) per dim (family scalar-equivalent init), b_k = 0.
Straight-through clamp on s_d in [1e-5, 100] as in the family.
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
from pretraining.fresh_lejepa.fresh_lejepa_train_v2_sigreg_projector import TokenProjector

HEAD_FORM = os.environ.get("ENERGY_HEAD_FORM", "distance")
assert HEAD_FORM in {"distance", "dot"}, HEAD_FORM
SCALE_MIN, SCALE_MAX = 1e-5, 100.0

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
    # Runs <=1992 steps see each token at most once, exactly as upstream.
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

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

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

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
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

class EnergyGPT(nn.Module):
    """nanogpt-mini trunk with the family's projected-codebook energy head."""

    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        # Registered after norm2 so shared trunk params keep the baseline's
        # named_parameters order (RNG-stream parity for init).
        self.latent_projector = TokenProjector(model_dim)
        self.prediction_projector = TokenProjector(model_dim)
        self.energy_log_scale = nn.Parameter(
            torch.full((model_dim,), -0.5 * math.log(model_dim), dtype=torch.float32)
        )
        self.energy_bias = nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))

    def energy_codebook(self) -> Tensor:
        raw = F.rms_norm(self.embed.weight, (self.embed.embedding_dim,))
        return self.latent_projector.inference(raw)

    def energy_logits(self, predicted: Tensor) -> Tensor:
        codebook = self.energy_codebook()
        log_scale = self.energy_log_scale
        # Straight-through elementwise clamp: forward sees the clamped value,
        # backward the unclamped gradient, so no dimension's bound is sticky.
        log_scale = log_scale + (
            log_scale.clamp(math.log(SCALE_MIN), math.log(SCALE_MAX)) - log_scale
        ).detach()
        scale = torch.exp(log_scale)  # (model_dim,), fp32
        scaled_predicted = predicted * scale.to(predicted.dtype)
        dots = (scaled_predicted @ codebook.transpose(0, 1)).float()
        if HEAD_FORM == "dot":
            return dots + self.energy_bias
        scale_f = scale.float()
        z_sq = (predicted.float().square() * scale_f).sum(dim=-1, keepdim=True)
        c_sq = (codebook.float().square() * scale_f).sum(dim=-1)
        return self.energy_bias - 0.5 * (z_sq - 2.0 * dots + c_sq)

    def forward(self, inputs: Tensor, targets: Tensor):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw = F.rms_norm(self.embed(inputs), (self.embed.embedding_dim,))
            token_latent = self.latent_projector(raw)
        x = self.norm1(token_latent.to(torch.bfloat16))
        for block in self.blocks:
            x = block(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = self.prediction_projector(self.norm2(x))
            logits = self.energy_logits(predicted)
        # No softcap: the energy head is capless by design (see docstring).
        return F.cross_entropy(logits.float().view(targets.numel(), -1),
                               targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
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
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


########################################
#                Setup                 #
########################################

# Single-process shim: default the torchrun env vars so `python3` launches work.
if "RANK" not in os.environ:
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29642")

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
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
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}"
       + f" head_form={HEAD_FORM}")
print0("="*100)

data_path = os.environ.get("DATA_PATH", "data/datasets/fineweb_onepass_sp1024")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "data/tokenizers/fineweb_1024_bpe.model")

val_tokens = 20 * 524288
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

model = EnergyGPT(vocab_size=1024, num_layers=6, model_dim=512).cuda()
model.compile(dynamic=False)
print0(f"parameters: {sum(p.numel() for p in model.parameters()):,}", console=True)


num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for _ in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    train_steps = int(os.environ.get("ITERATIONS", 1000))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 20))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))

    # seeded; shared trunk params consume RNG in the baseline's exact order
    torch.manual_seed(int(os.environ.get("SEED", 1337)))

    # initialize model parameters (baseline loop, plus the energy additions)
    for name, p in model.named_parameters():
        w = p.data
        if "_projector." in name or name.startswith("energy_"):
            continue  # handled below, after the shared-trunk RNG draws
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                # default torch init at std 1.0; EMBED_INIT_STD=0.005 gives the
                # family's tied_embed_init_std (same RNG draw count either way)
                w.normal_(std=float(os.environ.get("EMBED_INIT_STD", 1.0)))
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")
    # Family init for the new modules: reset_parameters() restores each
    # linear/BN default (what the family uses), deterministically post-seed.
    for module in (model.latent_projector, model.prediction_projector):
        for sub in module.modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()
    with torch.no_grad():
        model.energy_log_scale.fill_(-0.5 * math.log(model.embed.embedding_dim))
        model.energy_bias.zero_()

    # create the optimizer(s): upstream groups, minus the deleted proj group;
    # projector matrices join Muon (family routing), new ndim<2 params fall
    # into the existing scalar AdamW group automatically.
    # head scalars and the embed/codebook table get their own groups so the
    # family regime (EMBED_LR=0.05 EMBED_WD=0 HEAD_SCALAR_LR=0.04 HEAD_WD=0)
    # is reachable by env; defaults collapse to the upstream single-group math
    # (Adam is per-param, so an identical-hyperparam split changes nothing).
    head_scalars = [model.energy_log_scale, model.energy_bias]
    head_scalar_ids = {id(p) for p in head_scalars}
    other_scalars = [p for p in model.parameters()
                     if p.ndim < 2 and id(p) not in head_scalar_ids]
    optimizer1 = AdamW([dict(params=[model.embed.weight],
                             lr=float(os.environ.get("EMBED_LR", 0.7)),
                             weight_decay=float(os.environ.get("EMBED_WD", 0.001))),
                        dict(params=head_scalars,
                             lr=float(os.environ.get("HEAD_SCALAR_LR", 0.015)),
                             weight_decay=float(os.environ.get("HEAD_WD", 0.001))),
                        dict(params=other_scalars, lr=0.015)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2]
                      + [p for m in (model.latent_projector, model.prediction_projector)
                         for p in m.parameters() if p.ndim >= 2],
                      lr=0.025, weight_decay=0.05)
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta


    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator(f"{data_path}/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    # start the clock
    training_time = 0
    last_val_step = 0
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
            val_bpb = (float(val_loss) / math.log(2.0)) * (val_tokens / val_byte_count)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                scale = torch.exp(model.energy_log_scale.clamp(
                    math.log(SCALE_MIN), math.log(SCALE_MAX)))
                codebook = model.energy_codebook().float()
                cb_norm = codebook.norm(dim=-1).mean()
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} val_bpb:{val_bpb:.4f}"
                   + f" train_time:{1000*training_time:.0f}ms step_avg:{1000*step_avg:.2f}ms"
                   + f" scale_mean:{scale.mean():.4f} scale_max:{scale.max():.4f}"
                   + f" scale_min:{scale.min():.4f}"
                   + f" codebook_norm:{cb_norm:.2f}", console=True)
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
        train_loss_sum = torch.zeros((), device=device)
        for i in range(len(inputs) // mbs):
            loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
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
        approx_training_time = training_time + (time.perf_counter() - t0)
        if (step + 1) % train_log_every == 0:
            train_loss = float(train_loss_sum) / targets.numel()
            print0(f"step:{step+1}/{train_steps} train_loss:{train_loss:.4f}"
                   + f" train_time:{1000*approx_training_time:.0f}ms"
                   + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True)

dist.destroy_process_group()

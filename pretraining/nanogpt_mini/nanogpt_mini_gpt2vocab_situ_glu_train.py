"""pretraining/nanogpt_mini/nanogpt_mini_gpt2vocab_situ_glu_train.py

Vocab/tokenizer probe on the nanogpt-mini baseline (``pretraining/nanogpt_mini/nanogpt_mini_train.py``,
recipe kept VERBATIM): the 1024-token SentencePiece vocab is replaced by
nanogpt's native GPT-2 BPE vocab (50,304 padded rows; 50,257 real tokens),
training on modded-nanogpt's pre-tokenized fineweb10B GPT-2 shards.

SiTU-GLU ablation based on ``pretraining/nanogpt_mini/nanogpt_mini_gpt2vocab_train.py``. The ReLU
squared MLP is replaced by Kimi K3's SiTU-GLU with beta_gate=4 and
beta_up=25. A 1,368-wide bias-free gated MLP has 2,101,248 parameters per
block, within 0.08% of the 2,099,712 parameters in the baseline's
2,048-wide MLP, while retaining a hardware-friendly width divisible by
eight.

Question isolated: is the tiny 1024 vocab what caps the mini recipe's BPB?
BPB is byte-normalized, so the number is directly comparable to the sp1024
runs (same FineWeb val text distribution). At 524,288 tokens/step a GPT-2
token covers ~4.5 bytes vs sp1024's ~2.6, so a matched-STEP run sees ~1.7x
the text bytes — this is inherent to the vocab change, not a confound we
can remove; interpret wins accordingly.

DIAGNOSTIC ONLY, not submittable: embed + untied head at 50,304 x 512 is
~51.5M params — over the 16MB artifact budget by itself. If this wins big,
the actionable direction is intermediate vocabs (we already have sp8192
shards; the challenge SOTA submission uses 8192).

Changes vs pretraining/nanogpt_mini/nanogpt_mini_train.py (everything else verbatim):
  - vocab_size 50304; data default ``data/datasets/fineweb10B_gpt2``
    (symlink to ../modded-nanogpt/data/fineweb10B)
  - BPB byte accounting via ``data/tokenizers/gpt2_byte_lut.pt`` (per-token
    byte lengths from the GPT-2 byte-level BPE map; direct LUT-sum — GPT-2
    tokens map to fixed byte strings, so no leading-space logic needed;
    <|endoftext|> counted as 1 byte, a document-separator convention worth
    <0.1% of the denominator)
  - VAL_TOKENS default 33,554,432 (~150M bytes, ≈ the byte volume of the
    sp1024 full-val window's 62M tokens)
  - MBS env knob, default 8: logits are B*1024*50304 fp32, so the sp1024
    scripts' mbs=64 would OOM; 8 keeps the softcap+CE peak ~8GB
  - EMBED_LR (0.7) / PROJ_LR (0.004) env knobs: the mini's embed LR was
    tuned on a 1024-row table with dense row updates; 50k rows see ~50x
    sparser updates, so these are first-probe knobs
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

VOCAB_SIZE = 50304  # GPT-2 BPE (50,257 real tokens, padded for GPU efficiency)


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
    # cycle so runs longer than the downloaded shard slice keep training.
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

class BiasFreeLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), None)

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
        hdim = 1368
        self.gate = BiasFreeLinear(dim, hdim)
        self.up = BiasFreeLinear(dim, hdim)
        self.proj = BiasFreeLinear(hdim, dim)

    def forward(self, x: Tensor):
        gate = self.gate(x)
        up = self.up(x)
        capped_gate = 4.0 * torch.tanh(gate / 4.0)
        capped_up = 25.0 * torch.tanh(up / 25.0)
        return self.proj((capped_gate * torch.sigmoid(gate)) * capped_up)

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
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


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
    os.environ.setdefault("MASTER_PORT", "29647")

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
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0("="*100)

data_path = os.environ.get("DATA_PATH", "data/datasets/fineweb10B_gpt2")

val_tokens = int(os.environ.get("VAL_TOKENS", 64 * 524288))
batch_size = 8 * 64 * 1024
# SEQ_LEN reshapes the same token budget into longer rows (RoPE positions seen
# in pretraining bound the usable RL context). Halve MBS when doubling SEQ_LEN
# to keep microbatch tokens (and the 50304-wide logit buffer) constant.
seq_len = int(os.environ.get("SEQ_LEN", 1024))
mbs = int(os.environ.get("MBS", 8))
assert batch_size % (mbs * seq_len) == 0 and val_tokens % (mbs * seq_len) == 0
val_inputs, val_targets = next(distributed_data_generator(
    f"{data_path}/fineweb_val_*.bin", val_tokens, seq_len=seq_len))

# Challenge BPB metric with GPT-2 byte accounting: each GPT-2 BPE token maps
# to a fixed byte string, so a per-token byte-length LUT is exact.
gpt2_byte_lut = torch.load("data/tokenizers/gpt2_byte_lut.pt", weights_only=True).to(device)
assert gpt2_byte_lut.numel() == VOCAB_SIZE
with torch.no_grad():
    val_byte_count = float(gpt2_byte_lut[val_targets.reshape(-1)].to(torch.int64).sum())
    assert val_byte_count > 0

model = GPT(vocab_size=VOCAB_SIZE, num_layers=6, model_dim=512).cuda()
model.compile(dynamic=False)
print0(f"parameters: {sum(p.numel() for p in model.parameters()):,}", console=True)
print0(f"val window: {val_tokens:,} tokens = {val_byte_count:,.0f} bytes "
       f"({val_byte_count/val_tokens:.3f} bytes/token)", console=True)


num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for trial in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    train_steps = int(os.environ.get("ITERATIONS", 1000))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 20))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))

    # seeded so variants can share init RNG for the trunk
    seed = int(os.environ.get("SEED", 1337))
    torch.manual_seed(seed)

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith(".mlp.gate.weight"):
            # Advance the global RNG exactly as the baseline's 2048x512 fc
            # initialization does. This keeps all shared parameters in later
            # blocks bit-identical to the existing seeded baseline.
            baseline_fc = torch.empty(
                (4 * w.size(-1), w.size(-1)),
                device=w.device,
                dtype=w.dtype,
            )
            baseline_fc.normal_(std=0.33**0.5 / w.size(-1)**0.5)
            w.copy_(baseline_fc[:w.size(0)])
        elif name.endswith(".mlp.up.weight"):
            # The up projection has no ReLU-squared counterpart. Initialize it
            # from an independent per-layer stream without perturbing the
            # baseline-compatible global RNG stream.
            layer_idx = int(name.split(".")[1])
            generator = torch.Generator(device=w.device)
            generator.manual_seed(seed + 10_000 + layer_idx)
            w.normal_(std=0.33**0.5 / w.size(-1)**0.5, generator=generator)
        elif name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # create the optimizer(s)
    embed_lr = float(os.environ.get("EMBED_LR", 0.7))
    proj_lr = float(os.environ.get("PROJ_LR", 0.004))
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=embed_lr),
                        dict(params=[model.proj.weight], lr=proj_lr),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
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

    train_loader = distributed_data_generator(
        f"{data_path}/fineweb_train_*.bin", batch_size, seq_len=seq_len)
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
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} val_bpb:{val_bpb:.4f}"
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

    # self-describing checkpoint for postraining/model_io.py (off the clock)
    if dist.get_rank() == 0:
        suffix = f"_trial{trial}" if num_trials > 1 else ""
        ckpt_path = f"logs/{run_id}{suffix}_final_model.pt"
        torch.save({
            "model": model.state_dict(),
            "model_config": dict(vocab_size=VOCAB_SIZE, num_layers=6, model_dim=512, mlp_hidden=1368),
            "architecture": "nanogpt_mini_gpt2vocab_situ_glu_v1",
            # RoPE positions seen in pretraining bound the usable RL context
            # (half-truncate rotary has no extrapolation).
            "train_seq_len": seq_len,
        }, ckpt_path)
        print0(f"saved checkpoint: {ckpt_path}", console=True)

dist.destroy_process_group()

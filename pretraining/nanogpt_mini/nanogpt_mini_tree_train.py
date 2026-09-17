"""pretraining/nanogpt_mini/nanogpt_mini_tree_train.py

Tree-pointer fork of the nanogpt-mini baseline
(``pretraining/nanogpt_mini/nanogpt_mini_train.py``, kept verbatim except for
the points below). Every layer's dense causal attention is replaced by
``TreePointerAttention`` from ``nanogpt_mini_tree_model.py``: ``TREE_K``
root-to-leaf descents per query over a binary tree of the context, each
decision reading the two children's pooled keys with a routing query; the
query attends its ``K`` leaf tokens plus the summary rows of every sibling
subtree it did not descend into (``K (log2 T + 1)`` rows, the whole context
at coarse resolution). One shared K/V head (pointer-GQA). See the model
docstring.

Diff against the baseline script:
  1. Model: ``TreeGPT`` (same 6L/512d trunk, MLP, head, softcap); K/V are a
     single 128-d head plus a 128-d routing query ``qr``; the mixer adds the
     shared-depth controller of the iptr model (``head.ctrl`` 512 x K*R and
     its per-pointer vectors) and ``route_gains`` (K).
  2. Env knobs: ``TREE_K`` pointers per query (default 4), ``TREE_WIDTH``
     controller width (default 7), ``TREE_BIAS`` (default 0.5): a decision
     over a node of ``2^level`` tokens starts at logit ``+bias * level``
     toward the NEARER child (the iptr geometric locality prior in the
     tree's coordinates), ``TREE_BACKEND`` (``fused`` Triton, ``gather``
     reference).
  3. Init: ``head.out`` ~ N(0, 1/R); ``feat_in`` = 0; ``feat_out`` = 0 except
     the level slope; ``out_bias`` = 0; ``route_gains`` = 1. Controller
     vectors and ``route_gains`` are AdamW (vector group); ``head.ctrl`` and
     ``qr`` are block matrices and go to Muon.
  4. ``MBS`` default 16 (``SEQ_LEN`` 4096 is the intended setting: 16 rows x
     4096 = the baseline's 64 x 1024 tokens per microbatch).
  5. Checkpoint records ``architecture = "nanogpt_mini_tree_v1"``.

Deliberately unchanged: data, val protocol, optimizer split and LRs,
schedule, sum-CE loss, softcap, SDPA-equivalent score scale 0.12.
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

from pretraining.nanogpt_mini import nanogpt_mini_tree_model
from pretraining.nanogpt_mini.nanogpt_mini_tree_model import TreeGPT
with open(nanogpt_mini_tree_model.__file__) as f:
    model_code = f.read()  # logged alongside this file's code


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

# RMSNorm / Linear / Rotary / CausalSelfAttention / MLP / Block / GPT moved
# verbatim to pretraining/nanogpt_mini/nanogpt_mini_model.py (imported above) so postraining can build
# the architecture without executing this script's module-scope training run.


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
    os.environ.setdefault("MASTER_PORT", "29656")

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

# we begin by logging this file itself, then the shared model module
print0(code)
print0("="*100)
print0(model_code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0("="*100)

data_path = os.environ.get("DATA_PATH", "data/datasets/fineweb_onepass_sp1024")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "data/tokenizers/fineweb_1024_bpe.model")
vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))  # must match tokenizer/data

val_tokens = int(os.environ.get("VAL_TOKENS", 20 * 524288))
batch_size = 8 * 64 * 1024
# Packed-window length. Longer windows (e.g. 4096 for post-training RL bases)
# keep the same token budget per step; scale MBS down to hold microbatch
# tokens constant (64 rows @ 1024 == 16 rows @ 4096).
seq_len = int(os.environ.get("SEQ_LEN", 1024))
mbs = int(os.environ.get("MBS", 16))
tree_k = int(os.environ.get("TREE_K", 4))
tree_width = int(os.environ.get("TREE_WIDTH", 7))
tree_bias = float(os.environ.get("TREE_BIAS", 0.5))
tree_backend = os.environ.get("TREE_BACKEND", "fused")
# always-on local window of real tokens (prior mass 1); the tree covers the far context
tree_local = int(os.environ.get("TREE_LOCAL", 64))
assert tree_k > 0 and tree_width > 0
assert batch_size % seq_len == 0 and (batch_size // seq_len) % mbs == 0
val_inputs, val_targets = next(
    distributed_data_generator(f"{data_path}/fineweb_val_*.bin", val_tokens, seq_len=seq_len)
)

# Challenge BPB metric: fixed byte count of the val window via the same
# sentencepiece byte-LUT accounting used by train_gpt.py's eval.
sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = train_gpt.build_sentencepiece_luts(
    sp, vocab_size=vocab_size, device=device
)
with torch.no_grad():
    _prev = val_inputs.reshape(-1).to(torch.int64)
    _tgt = val_targets.reshape(-1)
    _bytes = base_bytes_lut[_tgt].to(torch.int64)
    _bytes += (has_leading_space_lut[_tgt] & ~is_boundary_token_lut[_prev]).to(torch.int64)
    val_byte_count = float(_bytes.sum())
    assert val_byte_count > 0

model = TreeGPT(vocab_size=vocab_size, num_layers=6, model_dim=512, seq_len=seq_len,
                num_pointers=tree_k, width=tree_width, backend=tree_backend,
                local_window=tree_local).cuda()
model.compile(dynamic=False)
print0(f"parameters: {sum(p.numel() for p in model.parameters()):,}", console=True)
print0(f"tree pointer attention: K={tree_k} depth={model.blocks[0].attn.depths} local={tree_local} "
       f"rows/query={model.blocks[0].attn.candidates_per_query} width={tree_width} bias={tree_bias} "
       f"backend={tree_backend} mbs={mbs} seq_len={seq_len}", console=True)


num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for trial in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    train_steps = int(os.environ.get("ITERATIONS", 1000))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 20))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))

    # seeded so the energy-head variant can share init RNG for its trunk
    torch.manual_seed(int(os.environ.get("SEED", 1337)))

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("head.out"):
            w.normal_(std=tree_width**-0.5)  # O(1) bit logits from step 0
        elif name.endswith("head.feat_out"):
            # locality prior: logit = +bias * level toward the nearer child
            w.zero_()
            w[..., nanogpt_mini_tree_model.LEVEL_FEATURE] = tree_bias
        elif name.endswith("head.feat_in"):
            w.zero_()
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # create the optimizer(s): the controller's vectors/scalars ([H, K, R] /
    # [H, K] storage but per-pointer vectors) and route_gains are AdamW with
    # the vector group; head.ctrl.weight and qr.weight are block matrices
    # and go to Muon with the rest.
    controller_tables = [p for name, p in model.named_parameters()
                         if (".head." in name and not name.endswith(("ctrl.weight", "ctrl.bias")))
                         or name.endswith("route_gains")]
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.7),
                        dict(params=[model.proj.weight], lr=0.004),
                        dict(params=[p for p in model.parameters() if p.ndim < 2] + controller_tables,
                             lr=0.015)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
    controller_ids = {id(p) for p in controller_tables}
    optimizer2 = Muon([p for p in model.blocks.parameters()
                       if p.ndim >= 2 and id(p) not in controller_ids],
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
        f"{data_path}/fineweb_train_*.bin", batch_size, seq_len=seq_len
    )
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

    # self-describing checkpoint (off the clock); not a postraining backbone
    if dist.get_rank() == 0:
        suffix = f"_trial{trial}" if num_trials > 1 else ""
        ckpt_path = f"logs/{run_id}{suffix}_final_model.pt"
        torch.save({
            "model": model.state_dict(),
            "model_config": dict(vocab_size=vocab_size, num_layers=6, model_dim=512, seq_len=seq_len,
                                 num_pointers=tree_k, width=tree_width, mlp_hidden=2048,
                                 backend=tree_backend, local_window=tree_local),
            "architecture": "nanogpt_mini_tree_v1",
            # RoPE positions actually seen in training; post-training derives
            # its context budget from this (no extrapolation on half-truncate).
            "train_seq_len": seq_len,
        }, ckpt_path)
        print0(f"saved checkpoint: {ckpt_path}", console=True)

dist.destroy_process_group()

"""pretraining/nanogpt_mini/nanogpt_mini_iptr_train.py

Interval-pointer sparse-attention fork of the nanogpt-mini baseline
(``pretraining/nanogpt_mini/nanogpt_mini_train.py``, kept verbatim except for
the points below). Every layer's dense causal attention is replaced by
``IntervalPointerAttention`` from ``nanogpt_mini_iptr_model.py``: per head,
``PTR_K`` stochastic pointers pick one of the ``M_t = ceil((t+1)/PTR_BLOCK)``
blocks that actually exist behind the query by decoding their bits as a
binary search over ``[0, M_t)`` (bit d halves the current interval; nearer
half on 0). Every leaf is a real block, so no sample is wasted on windows
before position 0, and a query with one block makes zero decisions. The
query attends to the sampled block plus the blocks reached by flipping one
consumed bit (``<= K * (ceil(log2 M_t) + 1) * PTR_BLOCK`` keys per query).
Every bit is sampled independently, all depths in parallel, from a
shared-depth controller ``w . silu(W x_t + V f) + v . f + c`` of width
``IPTR_WIDTH`` per (head, pointer), conditioned on routing features ``f``
(which bit, available context, jump scale of the bit). No dense layer
remains. See the model docstring.

Diff against the baseline script:
  1. Model: ``IPtrGPT`` (same 6L/512d/4-head trunk, MLP, head, softcap); the
     mixer adds the controller: ``ctrl`` (``dim x H*K*R`` = 512 x 112 at the
     defaults, same size as the position model's ``ptr``) plus the tiny
     per-(head, pointer) feature maps ``feat_in`` (F x R), ``feat_out`` (F),
     read-out ``out`` (R) and scalar ``out_bias``.
  2. Env knobs: ``PTR_K`` pointers per head (default 4), ``PTR_BLOCK`` window
     tokens (default 8), ``IPTR_WIDTH`` controller width (default 7 = the
     position model's bit count, so ``ctrl`` matches ``ptr`` in size),
     ``PTR_LOCAL`` always-on local window in tokens (default 0),
     ``IPTR_BIAS`` (default 0.5): the decision at depth ``d`` starts at
     logit ``-IPTR_BIAS * log2(jump_d)`` where ``jump_d = M_t / 2^(d+1)``
     blocks, via ``feat_out``'s jump-scale entry and ``out_bias`` -- the
     position model's geometric prior ``-slope * b`` for the ``2^b``-block
     bit, in absolute blocks (at a power-of-two block count the two address
     spaces coincide, so the priors match term for term; depth 0 is the
     LARGEST jump, so a slope on the depth index would be backwards).
     ``IPTR_PRIOR`` (default ``score``): how ``log pi`` enters the attention
     score. ``score`` adds it (the controller's counterfactual gradient is
     weighted by the counterfactual's own prior mass, so confident bits
     suppress their revision signal); ``straight`` adds ``log pi - sg(log
     pi)`` (content-only forward, unsaturating backward: measured to run
     away); ``leg`` adds ``pi(flip) - sg(pi(flip))`` per flip candidate:
     content-only forward, ``sigma'(z)``-weighted local-expectation
     gradient in the backward, ``pi`` a pure sampling distribution.
  3. Init: ``ctrl.weight`` takes the baseline's block-matrix draw (Muon);
     ``out`` ~ N(0, 1/R) so the controller emits O(1) logits from step 0;
     ``feat_in`` = 0 and ``feat_out`` = 0 except the jump-scale slope
     above; ``out_bias`` = ``IPTR_BIAS`` (so a 1-block jump is 50/50).
  4. Optimizer split: ``ctrl.weight`` joins the Muon block group; the
     controller vectors/scalars join the AdamW ``ndim < 2`` group (lr 0.015)
     even though ``out`` etc. are stored ``[H, K, R]`` — they are per-pointer
     vectors, not matrices, so orthogonalizing them is meaningless.
  5. ``PTR_BACKEND`` (default ``flex``): fused flex_attention at the baseline
     ``MBS`` 64 and the baseline whole-model compile; ``gather`` is the
     explicit-candidate reference (``MBS`` default 4; tests only).
  6. Checkpoint records ``architecture = "nanogpt_mini_iptr_v1"``.

Deliberately unchanged: data, val protocol, optimizer LRs, schedule, sum-CE
loss, softcap, SDPA-equivalent score scale 0.12.
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

from pretraining.nanogpt_mini import nanogpt_mini_iptr_model
from pretraining.nanogpt_mini.nanogpt_mini_iptr_model import IPtrGPT
with open(nanogpt_mini_iptr_model.__file__) as f:
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
    os.environ.setdefault("MASTER_PORT", "29655")

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
ptr_backend = os.environ.get("PTR_BACKEND", "flex")
# flex saves only q/k/v + logsumexp like the baseline's SDPA, so the baseline
# microbatch fits; the gather backend materializes candidate K/V per layer
# and needs ~16x fewer rows.
mbs = int(os.environ.get("MBS", 64 if ptr_backend == "flex" else 4))
ptr_k = int(os.environ.get("PTR_K", 4))
ptr_block = int(os.environ.get("PTR_BLOCK", 8))
iptr_width = int(os.environ.get("IPTR_WIDTH", 7))
iptr_bias = float(os.environ.get("IPTR_BIAS", 0.5))
# DIAGNOSTIC: ``linear`` swaps the shared-depth controller for independent
# per-depth linear logits (the position model's head) to measure what the
# sharing costs; it cannot serve contexts beyond the training length.
iptr_head = os.environ.get("IPTR_HEAD", "shared")
# DIAGNOSTIC: ``padded`` decodes over the fixed [0, 2^D) range and drops
# leaves beyond the query (the position model's addressing) to measure what
# the t-relative interval map costs. ``sequential`` samples bit d given the
# interval reached by bits < d (tree-structured address distribution; same
# shared head, features gain the interval's exact size and start).
iptr_decode = os.environ.get("IPTR_DECODE", "interval")
iptr_prior = os.environ.get("IPTR_PRIOR", "score")
# ``pooled``: the routing decision also reads the keys -- q against the
# mean pre-rotary key of each half of the current interval (prefix sums,
# 1/PTR_BLOCK of dense attention); needs IPTR_DECODE=sequential. ``position``:
# x_t and position features only.
iptr_route = os.environ.get("IPTR_ROUTE", "position")
# Always-on local window (tokens, multiple of PTR_BLOCK; 0 = pure pointer
# attention).
ptr_local = int(os.environ.get("PTR_LOCAL", 0))
assert ptr_k > 0 and iptr_width > 0
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

model = IPtrGPT(vocab_size=vocab_size, num_layers=6, model_dim=512,
                num_pointers=ptr_k, block_size=ptr_block, width=iptr_width,
                backend=ptr_backend, local_window=ptr_local, head=iptr_head,
                decode=iptr_decode, prior=iptr_prior, route=iptr_route).cuda()
# flex_attend is a torch.compiler.disable boundary (see its docstring), so
# the baseline's whole-model compile is safe for both training and eval.
model.compile(dynamic=False)
iptr_depths = nanogpt_mini_iptr_model.num_depths(seq_len, ptr_block)
print0(f"parameters: {sum(p.numel() for p in model.parameters()):,}", console=True)
print0(f"interval pointer attention: backend={ptr_backend} head={iptr_head} decode={iptr_decode} prior={iptr_prior} route={iptr_route} K={ptr_k} block={ptr_block} depths={iptr_depths} "
       f"width={iptr_width} local={ptr_local} max keys/query={model.blocks[0].attn.candidates_per_query(seq_len)} "
       f"bias={iptr_bias} mbs={mbs}", console=True)


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
            w.normal_(std=iptr_width**-0.5)  # O(1) bit logits from step 0
        elif name.endswith("head.feat_out"):
            # geometric prior over jump size: logit_d = -slope * (scale_d - 1)
            # = -slope * log2(blocks jumped), as the position model's -slope * b.
            # Padded (diagnostic) decode jumps 2^(D-1-d) blocks at every t, so
            # its prior goes on the depth feature instead: -slope * (D-1-d).
            w.zero_()
            if iptr_decode == "padded":
                w[..., nanogpt_mini_iptr_model.DEPTH_FEATURE] = iptr_bias
            else:
                w[..., nanogpt_mini_iptr_model.SCALE_FEATURE] = -iptr_bias
        elif name.endswith("head.out_bias"):
            w.fill_(-iptr_bias * (iptr_depths - 1) if iptr_decode == "padded" else iptr_bias)
        elif name.endswith("head.feat_in"):
            w.zero_()
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # create the optimizer(s)
    # controller vectors/scalars are stored [H, K, R] / [H, K] but are not
    # matrices: they take the AdamW vector group, not Muon. route_gains
    # ([H, K], init 1 by the ``gains`` rule) likewise.
    ctrl_vectors = [p for name, p in model.named_parameters()
                    if (".head." in name and not name.endswith("ctrl.weight")) or name.endswith("route_gains")]
    ctrl_ids = set(map(id, ctrl_vectors))
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.7),
                        dict(params=[model.proj.weight], lr=0.004),
                        dict(params=[p for p in model.parameters() if p.ndim < 2 and id(p) not in ctrl_ids]
                             + ctrl_vectors, lr=0.015)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2 and id(p) not in ctrl_ids],
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
            "model_config": dict(vocab_size=vocab_size, num_layers=6, model_dim=512,
                                 num_pointers=ptr_k, block_size=ptr_block, width=iptr_width,
                                 mlp_hidden=2048, backend=ptr_backend, local_window=ptr_local,
                                 head=iptr_head, decode=iptr_decode, prior=iptr_prior, route=iptr_route),
            "architecture": "nanogpt_mini_iptr_v1",
            # RoPE positions actually seen in training; post-training derives
            # its context budget from this (no extrapolation on half-truncate).
            "train_seq_len": seq_len,
        }, ckpt_path)
        print0(f"saved checkpoint: {ckpt_path}", console=True)

dist.destroy_process_group()

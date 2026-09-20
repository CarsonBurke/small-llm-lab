"""pretraining/nanogpt_mini/nanogpt_mini_feedback_train.py

Latent-feedback ("temporal residual") fork of the nanogpt-mini baseline
(``pretraining/nanogpt_mini/nanogpt_mini_train.py``, kept verbatim except
for the points below). The model is ``FeedbackGPT`` from
``nanogpt_mini_feedback_model.py``: the baseline trunk plus one of three
ways the top-layer state of earlier positions re-enters the stack
(``glu``: the full-bandwidth transformer's gated fusion, arXiv:2608.08888;
``add``: additive fusion with a zero-initialised map; ``lam``: a decayed
linear-attention memory over all earlier top states read at selected layers),
trained with Jacobi passes (pass k carries pass k-1's shifted top states)
and scored with the next-token loss on every pass.

Diff against the baseline script:
  1. Model: ``FeedbackGPT`` (same 6L/512d trunk, head, softcap). Env knobs:
     ``FB_MODE`` (none|glu|add|lam, default none), ``FB_PASSES`` (training
     passes, default 2; forced to 1 for ``none``), ``FB_DETACH`` (0/1: stop
     the gradient through the carried state), ``FB_NOISE`` (uniform jitter
     on the carried state in training, default 0.02), ``FB_WINDOW``
     (sliding attention window in tokens, 0 = full causal), ``FB_SEQ_TOKENS``
     (validation tokens for the sequential evaluation, default 262144),
     ``FB_SEQ_EVERY`` (run it every N validations, default 0 = final only).
     ``FB_MEMORY_LAYERS`` (LAM read sites: all, or comma-separated zero-based
     layer indices such as 0 or 0,2,4; the backbone always keeps all six layers).
  2. Validation prints the headline ``val_bpb`` for pass ``FB_PASSES``
     (the training regime; pass 1 for ``none``) and the extras
     ``val_bpb_p1`` .. ``val_bpb_p3`` (Jacobi passes on the full panel);
     at the final step (and every ``FB_SEQ_EVERY`` validations) also
     ``val_bpb_p8`` and ``val_bpb_seq``, the true recurrence evaluated token
     by token with a KV cache on the first ``FB_SEQ_TOKENS`` tokens, plus
     ``val_bpb_p1_seq``: pass 1 on that same subset so the two are
     comparable. Every mode runs the sequential path (for ``none`` it must
     agree with pass 1, a runtime check of the KV-cache evaluator).
  3. Optimizer split: feedback matrices (fusion maps, memory projections)
     join the Muon block group; the memory's decay logits and feature
     temperatures form a small AdamW group (lr 0.002, no decay).
  4. Init: names containing ``proj`` are zero-initialised as in the
     baseline, which zeroes the additive fusion map and the memory output
     projections (``add`` and ``lam`` start as the baseline); the GLU maps
     draw the baseline's matrix init; decays/temperatures are reset by
     ``reset_extra_parameters``.
  5. Checkpoint records ``architecture = "nanogpt_mini_feedback_v1"`` and
     the feedback configuration.

Deliberately unchanged: data, val protocol, optimizer LRs, schedule,
sum-CE loss, softcap, SDPA scale 0.12. No prefix mixin: every position
beyond the first is fused in the feedback passes, since the evaluation
regime is the full recurrence from position 0.
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

from pretraining.nanogpt_mini import nanogpt_mini_feedback_model, nanogpt_mini_model
from pretraining.nanogpt_mini.nanogpt_mini_feedback_model import FeedbackGPT
with open(nanogpt_mini_model.__file__) as f:
    model_code = f.read()  # logged alongside this file's code
with open(nanogpt_mini_feedback_model.__file__) as f:
    model_code += "\n" + "=" * 100 + "\n" + f.read()


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
    os.environ.setdefault("MASTER_PORT", "29641")

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
mbs = int(os.environ.get("MBS", 64))
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
    val_bytes_per_row = _bytes.view(-1, seq_len).sum(-1)

fb_mode = os.environ.get("FB_MODE", "none")
fb_passes = int(os.environ.get("FB_PASSES", 2)) if fb_mode not in ("none", "top") else 1
fb_detach = bool(int(os.environ.get("FB_DETACH", 0)))
fb_noise = float(os.environ.get("FB_NOISE", 0.02))
fb_window = int(os.environ.get("FB_WINDOW", 0))
fb_seq_tokens = int(os.environ.get("FB_SEQ_TOKENS", 262144))
fb_seq_every = int(os.environ.get("FB_SEQ_EVERY", 0))
fb_memory_layer_spec = os.environ.get("FB_MEMORY_LAYERS", "all")
fb_memory_layers = None if fb_memory_layer_spec == "all" else tuple(
    int(index) for index in fb_memory_layer_spec.split(",")
)
assert fb_passes >= 1 and fb_seq_tokens % seq_len == 0
assert fb_mode in ("none", "top") or fb_passes >= 2, "a Jacobi feedback mode needs a second pass to train its channel"
model = FeedbackGPT(vocab_size=vocab_size, num_layers=6, model_dim=512, mode=fb_mode,
                    detach=fb_detach, noise=fb_noise, attn_window=fb_window,
                    memory_layers=fb_memory_layers).cuda()
model.compile(dynamic=False, fullgraph=True)
print0(f"parameters: {sum(p.numel() for p in model.parameters()):,}", console=True)
print0(f"feedback: mode={fb_mode} passes={fb_passes} detach={fb_detach} noise={fb_noise} window={fb_window}",
       console=True)
if fb_mode in ("lam", "top"):
    print0(f"memory: kernel={nanogpt_mini_feedback_model.MEMORY_KERNEL}"
           f" compiled={nanogpt_mini_feedback_model.MEMORY_COMPILED}"
           f" layers={model.config.get('memory_layers')}", console=True)


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
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        elif name in ("decay_logit", "temperature"):
            pass  # reset below from the model's own contract
        else:
            raise Exception(f"Uninitialized parameter: {name}")
    model.reset_extra_parameters()

    # create the optimizer(s)
    extra_scalars = model.extra_scalar_parameters()
    extra_ids = {id(p) for p in extra_scalars}
    adam_groups = [dict(params=[model.embed.weight], lr=0.7),
                   dict(params=[model.proj.weight], lr=0.004),
                   dict(params=[p for p in model.parameters() if p.ndim < 2 and id(p) not in extra_ids], lr=0.015)]
    if extra_scalars:
        adam_groups.append(dict(params=extra_scalars, lr=0.002, weight_decay=0.0))
    optimizer1 = AdamW(adam_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2] + model.feedback_matrices(),
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
        if step == train_steps and dist.get_rank() == 0:
            # self-describing checkpoint for postraining/model_io.py, written before the
            # final (deep) validation so an evaluator failure cannot lose the weights
            suffix = f"_trial{trial}" if num_trials > 1 else ""
            ckpt_path = f"logs/{run_id}{suffix}_final_model.pt"
            torch.save({
                "model": model.state_dict(),
                "model_config": model.config,
                "architecture": "nanogpt_mini_feedback_v1",
                # RoPE positions actually seen in training; post-training derives
                # its context budget from this (no extrapolation on half-truncate).
                "train_seq_len": seq_len,
            }, ckpt_path)
            print0(f"saved checkpoint: {ckpt_path}", console=True)

        if step == train_steps or step % val_loss_every == 0:
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            final = step == train_steps
            # The sequential recurrence is scored at step 0 (exercises the path before any
            # compute is spent), at the end, and every FB_SEQ_EVERY validations if set.
            deep = final or step == 0 or (fb_seq_every > 0 and (step // val_loss_every) % fb_seq_every == 0)
            eval_passes = max(3, fb_passes) if fb_mode not in ("none", "top") else 1
            if deep and fb_mode not in ("none", "top"):
                eval_passes = max(eval_passes, 8)
            pass_sums = torch.zeros(eval_passes, device=device)
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    losses = model.pass_losses(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs], eval_passes)
                    pass_sums += torch.stack(losses)
            dist.all_reduce(pass_sums, op=dist.ReduceOp.SUM)
            pass_losses = (pass_sums / val_tokens).tolist()
            to_bpb = lambda loss, tokens, bytes_: (loss / math.log(2.0)) * (tokens / bytes_)
            headline = pass_losses[fb_passes - 1]
            val_loss, val_bpb = headline, to_bpb(headline, val_tokens, val_byte_count)
            extras = "".join(f" val_bpb_p{k + 1}:{to_bpb(l, val_tokens, val_byte_count):.4f}"
                             for k, l in enumerate(pass_losses) if k + 1 in (1, 2, 3, 8))
            if fb_mode == "top":
                noread = torch.zeros((), device=device)
                with torch.no_grad():
                    for i in range(len(val_inputs) // mbs):
                        noread += model.loss_noread(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
                dist.all_reduce(noread, op=dist.ReduceOp.SUM)
                extras += f" val_bpb_noread:{to_bpb(float(noread) / val_tokens, val_tokens, val_byte_count):.4f}"
            if deep:
                seq_rows = fb_seq_tokens // seq_len
                seq_inputs, seq_targets = val_inputs[:seq_rows], val_targets[:seq_rows]
                seq_bytes = float(val_bytes_per_row[:seq_rows].sum())
                seq_sum = torch.zeros((), device=device)
                p1_sum = torch.zeros((), device=device)
                assert seq_rows <= len(val_inputs), (seq_rows, len(val_inputs))
                seq_mbs = min(mbs, seq_rows)
                with torch.no_grad():
                    for i in range(0, seq_rows, seq_mbs):
                        seq_sum += model.loss_sequential(seq_inputs[i:i + seq_mbs], seq_targets[i:i + seq_mbs])
                        p1_sum += model.pass_losses(seq_inputs[i:i + seq_mbs], seq_targets[i:i + seq_mbs], 1)[0]
                seq_stats = torch.stack((seq_sum, p1_sum, torch.tensor(seq_bytes, device=device)))
                dist.all_reduce(seq_stats, op=dist.ReduceOp.SUM)
                seq_sum, p1_sum, seq_bytes = (float(x) for x in seq_stats)
                seq_tokens = seq_rows * seq_len * dist.get_world_size()
                seq_bpb = to_bpb(seq_sum / seq_tokens, seq_tokens, seq_bytes)
                p1_seq_bpb = to_bpb(p1_sum / seq_tokens, seq_tokens, seq_bytes)
                extras += f" val_bpb_seq:{seq_bpb:.4f} val_bpb_p1_seq:{p1_seq_bpb:.4f}"
                # none: identical math; top: identical math up to the bf16 kernel read (no Jacobi gap).
                seq_gate = {"none": 0.002, "top": 0.005}.get(fb_mode)
                if seq_gate is not None and abs(seq_bpb - p1_seq_bpb) > seq_gate:
                    raise RuntimeError(f"sequential evaluator disagrees with the parallel forward: {seq_bpb} vs {p1_seq_bpb}")
            if fb_mode in ("lam", "top"):
                decays = model.decays.detach().tolist()
                extras += "".join(f" mem_decay{h}:{d:.4f}" for h, d in enumerate(decays))
                extras += "".join(f" mem_temp{h}:{t:.3f}" for h, t in enumerate(model.temperature.detach().tolist()))
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} val_bpb:{val_bpb:.4f}"
                   + f" train_time:{1000*training_time:.0f}ms step_avg:{1000*step_avg:.2f}ms" + extras, console=True)
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
            loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs], fb_passes)
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

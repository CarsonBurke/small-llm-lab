"""pretraining/nanogpt_mini/nanogpt_mini_slot_train.py

Slot-memory fork of the nanogpt-mini baseline
(``pretraining/nanogpt_mini/nanogpt_mini_train.py``, kept verbatim except for
the points below). Every layer attends over the token itself plus at most
``SLOT_M`` alive memory entries chosen by a learned write policy; see
``nanogpt_mini_slot_model.py`` for the mechanism, the Jacobi passes, and the
exact loss.

Diff against the baseline script:
  1. Model: ``SlotGPT`` (same 6L/512d trunk, MLP, head, softcap, registration
     order; ``full`` mode is bitwise the base ``GPT``). ``policy`` mode adds
     ``slot_head`` (fp32 ``Linear(512, M + 1)``, zero-init = uniform policy)
     and two baseline buffers; ``fifo``/``full`` add nothing.
  2. Env knobs (every one documented here):
       SLOT_MODE      policy | fifo | full (default policy)
       SLOT_M         slots per sequence (default 64)
       SLOT_PASSES    Jacobi passes in policy mode (default 2: full reference
                      pass + one slot-restricted pass; pass k uses the slots
                      sampled by pass k-1)
       SLOT_GAMMA     discount of the future-credit advantage (default 0.9)
       SLOT_HORIZON   horizon H of the advantage sum (default 32)
       SLOT_ENTROPY   entropy-bonus coefficient on the slot policy (default 0)
       SLOT_DETACH    1 = the slot term's gradient stops at the slot head and
                      never enters the trunk (default 0)
       SLOT_BASELINE_DECAY  EMA decay of the advantage baseline per
                      microbatch (default 0.99; the first microbatch seeds it)
       SLOT_SEQ_TOKENS      validation tokens of the final-step sequential
                      (true KV-cache) evaluation (default 262144; policy and
                      fifo modes, where the sequential regime differs from the
                      parallel replay in principle; full mode is causal SDPA,
                      whose sequential regime is the parallel pass by
                      construction and is skipped)
       SLOT_BACKEND   flex (production, compiled FlexAttention over a block
                      mask derived from the (T, M) table) | dense (explicit
                      [B, T, T] visibility through SDPA; reference/tests)
  3. Init: ``slot_head.*`` is zeroed (the baseline rules would draw a normal
     weight); it is registered after the trunk, so the trunk's seeded draws
     are those of the baseline. ``slot_head.weight`` (a matrix) joins the
     Muon group; ``slot_head.bias`` is in the AdamW ndim<2 group.
  4. Validation prints, besides ``val_loss``/``val_bpb`` (headline: the last
     slot-restricted pass; the only pass in fifo/full), ``val_bpb_full``
     (unrestricted pass), ``val_slot_null`` (fraction of null writes),
     ``val_slot_entropy`` (mean slot-policy entropy), ``val_slot_age`` (mean
     age of attended non-self entries, softmax-mass weighted, over layers).
     The final validation adds ``val_bpb_seq`` (sequential regime, sampled
     slots), ``val_bpb_seq_greedy`` (argmax slots; fifo: identical to
     ``val_bpb_seq``) and ``val_seq_null`` over the first SLOT_SEQ_TOKENS
     validation tokens. Training lines add ``train_ce_slot``,
     ``train_slot_null``, ``train_slot_entropy``, ``train_adv_mean``,
     ``train_adv_std``, ``train_baseline``, ``train_pg``, ``train_ce_full``
     (policy mode only: there ``train_loss`` is CE_1 + CE_2 + the policy
     term per token and is not comparable to the baseline's;
     ``train_ce_full``, the unrestricted pass, is).
  5. The training forward returns ``(loss, stats)``; ``loss`` is what the
     baseline would backward in ``full`` mode.
  6. Checkpoint records ``architecture = "nanogpt_mini_slot_v1"`` and the
     slot configuration in ``model_config``.

Deliberately unchanged: data, val window, optimizer split and LRs, schedule,
sum-CE loss, softcap, SDPA scale 0.12, MBS 64 x seq 1024, compile of the
training forward (the sequential evaluator is eager).
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

from pretraining.nanogpt_mini import nanogpt_mini_slot_model
from pretraining.nanogpt_mini.nanogpt_mini_slot_model import SlotGPT
with open(nanogpt_mini_slot_model.__file__) as f:
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
    os.environ.setdefault("MASTER_PORT", "29663")

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
slot_mode = os.environ.get("SLOT_MODE", "policy")
slot_m = int(os.environ.get("SLOT_M", 64))
slot_passes = int(os.environ.get("SLOT_PASSES", 2))
slot_gamma = float(os.environ.get("SLOT_GAMMA", 0.9))
slot_horizon = int(os.environ.get("SLOT_HORIZON", 32))
slot_entropy = float(os.environ.get("SLOT_ENTROPY", 0.0))
slot_detach = bool(int(os.environ.get("SLOT_DETACH", 0)))
slot_baseline_decay = float(os.environ.get("SLOT_BASELINE_DECAY", 0.99))
slot_seq_tokens = int(os.environ.get("SLOT_SEQ_TOKENS", 262144))
slot_backend = os.environ.get("SLOT_BACKEND", "flex")
assert slot_mode in nanogpt_mini_slot_model.MODES, slot_mode
assert slot_backend in nanogpt_mini_slot_model.BACKENDS, slot_backend
assert batch_size % seq_len == 0 and (batch_size // seq_len) % mbs == 0
assert slot_seq_tokens % seq_len == 0 and slot_seq_tokens <= val_tokens
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

model = SlotGPT(vocab_size=vocab_size, num_layers=6, model_dim=512, slots=slot_m, mode=slot_mode,
                passes=slot_passes, gamma=slot_gamma, horizon=slot_horizon, entropy_coef=slot_entropy,
                detach=slot_detach, backend=slot_backend, baseline_decay=slot_baseline_decay).cuda()
model.compile(dynamic=False)
print0(f"parameters: {sum(p.numel() for p in model.parameters()):,}", console=True)
print0(f"slot memory: mode={slot_mode} M={slot_m} passes={model.passes} gamma={slot_gamma} "
       f"horizon={slot_horizon} entropy={slot_entropy} detach={int(slot_detach)} "
       f"baseline_decay={slot_baseline_decay} seq_tokens={slot_seq_tokens} backend={slot_backend}", console=True)


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
        if name.startswith("slot_head."):
            w.zero_()  # uniform initial policy over M + 1 choices
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
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.7),
                        dict(params=[model.proj.weight], lr=0.004),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
    slot_matrices = [model.slot_head.weight] if slot_mode == "policy" else []
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2] + slot_matrices,
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
            val_sums = {k: torch.zeros((), device=device) for k in
                        ("ce_last", "ce_full", "null_count", "entropy_sum", "age_num", "age_den")}
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    _, stats = model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
                    for k in val_sums:
                        val_sums[k] += stats[k].float()
            for k in val_sums:
                dist.all_reduce(val_sums[k], op=dist.ReduceOp.SUM)
            to_bpb = lambda nats: (nats / math.log(2.0)) * (val_tokens / val_byte_count)
            val_loss = float(val_sums["ce_last"]) / val_tokens
            val_bpb = to_bpb(val_loss)
            val_bpb_full = to_bpb(float(val_sums["ce_full"]) / val_tokens)
            val_slot_null = float(val_sums["null_count"]) / val_tokens
            val_slot_entropy = float(val_sums["entropy_sum"]) / val_tokens
            val_slot_age = float(val_sums["age_num"]) / max(float(val_sums["age_den"]), 1e-12)
            extra = (f" val_bpb_full:{val_bpb_full:.4f} val_slot_null:{val_slot_null:.4f}"
                     f" val_slot_entropy:{val_slot_entropy:.4f} val_slot_age:{val_slot_age:.2f}")
            if step == train_steps and slot_mode != "full":
                # true sequential regime (eager, token by token with slot banks)
                seq_rows = slot_seq_tokens // seq_len
                seq_bytes = float(_bytes[:slot_seq_tokens].sum())
                # fifo has no policy: one deterministic pass stands for both numbers
                seq_choices = ("sample", "greedy") if slot_mode == "policy" else ("fifo",)
                seq_results = []
                with torch.no_grad():
                    for choice in seq_choices:
                        ce_sum = torch.zeros((), device=device)
                        null_sum = torch.zeros((), device=device)
                        for i in range(0, seq_rows, mbs):
                            out = model.sequential(val_inputs[i:i+mbs], val_targets[i:i+mbs], choice=choice)
                            ce_sum += out["ce"].sum()
                            null_sum += out["null_count"].float()
                        seq_results.append((float(ce_sum), float(null_sum)))
                if len(seq_results) == 1:
                    seq_results.append(seq_results[0])
                seq_bpb = [(nats / slot_seq_tokens / math.log(2.0)) * (slot_seq_tokens / seq_bytes)
                           for nats, _ in seq_results]
                extra += (f" val_bpb_seq:{seq_bpb[0]:.4f} val_bpb_seq_greedy:{seq_bpb[1]:.4f}"
                          f" val_seq_null:{seq_results[0][1] / slot_seq_tokens:.4f}")
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} val_bpb:{val_bpb:.4f}" + extra
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
        train_keys = ("ce_last", "ce_full", "null_count", "entropy_sum", "adv_mean", "adv_std", "baseline", "pg_sum")
        train_sums = {k: torch.zeros((), device=device) for k in train_keys}
        for i in range(len(inputs) // mbs):
            loss, stats = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
            train_loss_sum += loss.detach()
            for k in train_keys:
                if k in stats:
                    train_sums[k] += stats[k].float()
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
            n_micro = len(inputs) // mbs
            extra = (f" train_ce_slot:{float(train_sums['ce_last']) / targets.numel():.4f}"
                     f" train_slot_null:{float(train_sums['null_count']) / targets.numel():.4f}"
                     f" train_slot_entropy:{float(train_sums['entropy_sum']) / targets.numel():.4f}")
            if slot_mode == "policy":  # fifo trains no unrestricted pass; full's is train_loss itself
                extra += (f" train_ce_full:{float(train_sums['ce_full']) / targets.numel():.4f}"
                          f" train_adv_mean:{float(train_sums['adv_mean']) / n_micro:.4f}"
                          f" train_adv_std:{float(train_sums['adv_std']) / n_micro:.4f}"
                          f" train_baseline:{float(train_sums['baseline']) / n_micro:.4f}"
                          f" train_pg:{float(train_sums['pg_sum']) / targets.numel():.4f}")
            print0(f"step:{step+1}/{train_steps} train_loss:{train_loss:.4f}" + extra
                   + f" train_time:{1000*approx_training_time:.0f}ms"
                   + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True)

    # self-describing checkpoint (off the clock); not a postraining backbone
    if dist.get_rank() == 0:
        suffix = f"_trial{trial}" if num_trials > 1 else ""
        ckpt_path = f"logs/{run_id}{suffix}_final_model.pt"
        torch.save({
            "model": model.state_dict(),
            "model_config": dict(vocab_size=vocab_size, num_layers=6, model_dim=512, mlp_hidden=2048,
                                 slots=slot_m, mode=slot_mode, passes=model.passes, gamma=slot_gamma,
                                 horizon=slot_horizon, entropy_coef=slot_entropy, detach=slot_detach,
                                 backend=slot_backend, baseline_decay=slot_baseline_decay),
            "architecture": "nanogpt_mini_slot_v1",
            # RoPE positions actually seen in training; post-training derives
            # its context budget from this (no extrapolation on half-truncate).
            "train_seq_len": seq_len,
        }, ckpt_path)
        print0(f"saved checkpoint: {ckpt_path}", console=True)

dist.destroy_process_group()

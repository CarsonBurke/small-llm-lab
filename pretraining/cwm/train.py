"""Pretrain a Causal Workspace Machine arm on FineWeb (GPT-2 tokens).

Arms (``research_notes/Optimal causal model communication/cwm_plan.md``):

* ``--variant cwm``: causal softmax-attention perception layer, banks plus
  the exact windowed store;
* ``--variant kda``: KDA perception layer, banks only (no softmax attention
  anywhere);

each with ``--read-mask full`` (feedback reads), ``lattice`` (the matched
parallel-computable control), or ``placebo`` (full's feedback paths fed
another row's memory: the capacity control). All three are bitwise
identical at initialization.

``--lag feedback`` or ``--lag all`` trains a chunk-lagged contract
(``CWMConfig.lag``) on the chunk-parallel core (``parallel.py``), with fp32
banks and without TF32; ``--lag none`` (default) is the exact contract on
the fused core.

Recipe (the KDA8 trainer's): Muon 0.025 (per-head on the perception Q/K/V,
wd 0.05, momentum 0.85 -> 0.95 over 500 steps) on matrices; AdamW (0.8,
0.95), eps 1e-10, wd 0.001, with embedding 0.7, head 0.004, short-conv
0.004, scalars 0.015, and decay parameters 0.015 without weight decay;
stable-then-linear schedule with a 70% cooldown and no warmup; summed CE.

Validation scores a prefix of the KDA8 trainer's panel (the start of
``fineweb_val_*``, 1,024-token rows), with BPB from the GPT-2 byte LUT.

Outputs: ``ablation_results/<name>/{manifest.json,metrics.jsonl,source/,
final_model.pt,result.json}`` and TensorBoard events in ``tb_logs/<name>``.

Run through mlq, e.g.:

    mlq submit --name cwm_full_1k --cwd "$PWD" --max-parallel-runs 1 -- \\
      .venv/bin/python -m pretraining.cwm.train --name cwm_full_1k --variant cwm --read-mask full
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pretraining.cwm.graphed import GraphedCore, graphed_core
from pretraining.cwm.parallel import MAX_LAG_CHUNK, GraphedParallelCore, graphed_parallel_core
from pretraining.cwm.model import (
    ARCHITECTURE,
    CWM,
    CWMConfig,
    config_dict,
    initialize,
    parameter_report,
)
from pretraining.hbyte.data import ContiguousStream, file_sha256, shard_tokens
from pretraining.hbyte.train import MetricLog, Muon, component_grad_norms, git_revision, gpu_neighbors
from pretraining.nanogpt_mini.nanogpt_mini_kda_model import CausalSelfAttention, KimiDeltaAttention

LN2 = math.log(2)
REPO = Path(__file__).resolve().parents[2]
DATA = Path("data/datasets/fineweb10B_gpt2")
BYTE_LUT = Path("data/tokenizers/gpt2_byte_lut.pt")
CONTEXT_BINS = (0, 64, 256, 512)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def build_optimizers(model: CWM, args) -> list[torch.optim.Optimizer]:
    decay, conv, per_head = [], [], []
    for module in model.modules():
        if isinstance(module, KimiDeltaAttention):
            decay += [module.A_log, module.dt_bias]
            conv += [c.weight for c in (module.q_conv1d, module.k_conv1d, module.v_conv1d)]
            per_head += [(l.weight, module.num_heads) for l in (module.q_proj, module.k_proj, module.v_proj)]
        elif isinstance(module, CausalSelfAttention):
            per_head += [(l.weight, module.num_heads) for l in (module.q, module.k, module.v)]
    writers = [blk.writer for blk in model.rounds] + [model.out_writer]
    # Frozen parameters (the lattice arm's feedback paths) belong to no group.
    decay += [p for w in writers for p in (w.A_log, w.dt_bias) if p.requires_grad]
    lookup, head = [model.embed.weight], [model.head.weight]
    special = {id(p) for p in lookup + head + decay + conv} | {id(p) for p, _ in per_head}
    # Muon takes the weight matrices of linear maps; every other tensor
    # (biases, gains, tags) is a scalar-like parameter for AdamW.
    matrices = [
        (m.weight, 1)
        for m in model.modules()
        if isinstance(m, nn.Linear) and m.weight.requires_grad and id(m.weight) not in special
    ]
    matrix_ids = {id(p) for p, _ in matrices}
    scalars = [
        p for p in model.parameters() if p.requires_grad and id(p) not in special and id(p) not in matrix_ids
    ]
    adam = torch.optim.AdamW(
        [
            dict(params=lookup, lr=args.embed_lr),
            dict(params=head, lr=args.head_lr),
            dict(params=scalars, lr=args.scalar_lr),
            dict(params=conv, lr=args.conv_lr),
            dict(params=decay, lr=args.scalar_lr, weight_decay=0.0),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=args.adam_wd,
        fused=True,
    )
    muon = Muon(matrices + per_head, lr=args.muon_lr, weight_decay=args.muon_wd, mu=0.85)
    owned = [p for opt in (adam, muon) for g in opt.param_groups for p in g["params"]]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if len(owned) != len({id(p) for p in owned}) or {id(p) for p in owned} != trainable:
        raise RuntimeError("every trainable parameter must belong to exactly one optimizer group")
    for opt in (adam, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return [adam, muon]


def set_schedule(optimizers, step: int, steps: int, cooldown_frac: float = 0.7, momentum_warmup: int = 500) -> float:
    """Stable then linear decay to zero over the last ``cooldown_frac``."""
    progress = step / steps
    eta = 1.0 if progress < 1 - cooldown_frac else (1 - progress) / cooldown_frac
    mu = 0.85 + min(step / momentum_warmup, 1.0) * (0.95 - 0.85)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta
            if "mu" in group:
                group["mu"] = mu
    return eta


# ---------------------------------------------------------------------------
# Loss and evaluation
# ---------------------------------------------------------------------------


def head_nll(model: CWM, hidden: Tensor, targets: Tensor) -> Tensor:
    """Per-token NLL ``[rows, T]`` (fp32 logits, softcapped)."""
    logits = model.project_logits(hidden)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)


class Objective:
    """Summed next-token CE of one microbatch, with the vocabulary
    projection applied in row slices: full fp32 logits of a 128 x 1,024
    microbatch would be 26 GiB. The core runs once; its output gradient is
    accumulated slice by slice, then sent back through the core in one
    backward.

    The core runs on CUDA graphs, one set per batch size, grad mode and
    feedback mode, captured on first use: fused (``graphed.GraphedCore``)
    for the exact contract, chunk-parallel (``parallel.GraphedParallelCore``)
    for a lagged one. No autocast: activations are bf16 (the embedding's
    dtype), recurrent state and mixing fp32."""

    def __init__(self, model: CWM, args):
        self.model, self.args = model, args
        self.lagged = model.cfg.lag != "none"
        self.nll = torch.compile(lambda h, y: head_nll(model, h, y), dynamic=False)
        self.cores: dict[tuple[int, bool, object], GraphedCore | GraphedParallelCore] = {}

    def core(self, batch: int, train: bool) -> GraphedCore | GraphedParallelCore:
        key = (batch, train, self.model.feedback)
        if key not in self.cores:
            if self.lagged:
                self.cores[key] = GraphedParallelCore(self.model, batch, self.args.seq_len, train=train)
            else:
                self.cores[key] = GraphedCore(self.model, batch, self.args.seq_len, self.args.chunk, train=train)
        return self.cores[key]

    def hidden(self, inputs: Tensor, train: bool) -> Tensor:
        model = self.model
        u = model.perceive(inputs)
        run = graphed_parallel_core if self.lagged else graphed_core
        return model.norm2(run(self.core(inputs.size(0), train), u))

    def train(self, inputs: Tensor, targets: Tensor) -> Tensor:
        hidden = self.hidden(inputs, train=True)
        detached = hidden.detach().requires_grad_()
        total = torch.zeros((), device=inputs.device)
        for rows in torch.split(torch.arange(inputs.size(0), device=inputs.device), self.args.head_rows):
            nll = self.nll(detached[rows], targets[rows]).sum()
            nll.backward()
            total += nll.detach()
        hidden.backward(detached.grad)
        return total

    @torch.no_grad()
    def evaluate(self, inputs: Tensor, targets: Tensor) -> Tensor:
        hidden = self.hidden(inputs, train=False)
        return torch.cat([self.nll(h, y) for h, y in zip(hidden.split(self.args.head_rows), targets.split(self.args.head_rows))])


def validation_rows(tokens: int, seq_len: int) -> tuple[Tensor, Tensor]:
    """The KDA8 trainer's panel prefix: ``fineweb_val`` tokens ``[0, N]``."""
    stream = np.asarray(shard_tokens(sorted(DATA.glob("fineweb_val_*.bin"))[0])[: tokens + 1]).astype(np.int64)
    if stream.size != tokens + 1:
        raise ValueError("validation shard is shorter than the requested panel")
    stream = torch.from_numpy(stream)
    return stream[:-1].view(-1, seq_len), stream[1:].view(-1, seq_len)


def evaluate(objective: Objective, inputs: Tensor, targets: Tensor, lut: Tensor, microbatch: int) -> dict:
    model = objective.model
    model.eval()
    nll_sum = torch.zeros((), device="cuda", dtype=torch.float64)
    T = inputs.size(1)
    edges = torch.tensor(CONTEXT_BINS, device="cuda")
    bins = torch.bucketize(torch.arange(T, device="cuda"), edges, right=True) - 1
    ctx_nll = torch.zeros(len(CONTEXT_BINS), device="cuda", dtype=torch.float64)
    ctx_bytes = torch.zeros(len(CONTEXT_BINS), device="cuda", dtype=torch.float64)
    total_bytes = 0
    for start in range(0, inputs.size(0), microbatch):
        x = inputs[start : start + microbatch].cuda(non_blocking=True)
        y = targets[start : start + microbatch].cuda(non_blocking=True)
        nll = objective.evaluate(x, y).double()
        nbytes = lut[y].double()
        nll_sum += nll.sum()
        total_bytes += int(nbytes.sum())
        ctx_nll += torch.zeros_like(ctx_nll).index_add_(0, bins, nll.sum(0))
        ctx_bytes += torch.zeros_like(ctx_bytes).index_add_(0, bins, nbytes.sum(0))
    model.train()
    out = {"bpb": nll_sum.item() / LN2 / total_bytes, "tokens": inputs.numel(), "bytes": total_bytes}
    for i, lo in enumerate(CONTEXT_BINS):
        out[f"ctx_{lo}_bpb"] = (ctx_nll[i] / ctx_bytes[i]).item() / LN2
    return out


def channel_norms(model: CWM) -> dict:
    """Frobenius norms of each round's read projections: the feedback
    projections start at zero, so their growth is the channel opening."""
    out = {}
    for r, blk in enumerate(model.rounds, start=1):
        names = ["bank_o_legal", "bank_o_fb", "depth_o"] + (["store_o_legal", "store_o_fb"] if model.cfg.store else [])
        for name in names:
            out[f"r{r}_{name}"] = getattr(blk, name).weight.float().norm().item()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--variant", choices=("cwm", "kda"), required=True)
    p.add_argument("--read-mask", choices=("full", "lattice", "placebo"), required=True)
    p.add_argument("--bank-dtype", choices=("float32", "bfloat16"), default=None,
                   help="bank state storage (math is fp32); see CWMConfig.bank_dtype. "
                   "Default: bfloat16 for the exact contract, float32 (required) under lag")
    p.add_argument("--lag", choices=("none", "feedback", "all"), default="none", help="see CWMConfig.lag")
    p.add_argument("--lag-chunk", type=int, default=16, help="tokens per lag chunk (CWMConfig.lag_chunk)")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--rows-per-step", type=int, default=512, help="512 x 1,024 = 524,288 tokens, the KDA8 batch")
    p.add_argument("--microbatch", type=int, default=128)
    p.add_argument("--chunk", type=int, default=16, help="tokens per checkpointed core segment (exact contract)")
    p.add_argument("--head-rows", type=int, default=8, help="rows per vocabulary-projection slice")
    p.add_argument("--val-tokens", type=int, default=2_097_152)
    p.add_argument("--final-val-tokens", type=int, default=8_388_608)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--eval-microbatch", type=int, default=256)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--embed-lr", type=float, default=0.7)
    p.add_argument("--head-lr", type=float, default=0.004)
    p.add_argument("--scalar-lr", type=float, default=0.015)
    p.add_argument("--conv-lr", type=float, default=0.004)
    p.add_argument("--muon-lr", type=float, default=0.025)
    p.add_argument("--adam-wd", type=float, default=0.001)
    p.add_argument("--muon-wd", type=float, default=0.05)
    p.add_argument("--results-root", type=Path, default=REPO)
    p.add_argument("--no-save", action="store_true", help="timing runs: skip the final checkpoint")
    args = p.parse_args()
    if args.rows_per_step % args.microbatch:
        raise SystemExit("rows-per-step must be a multiple of microbatch")
    if args.seq_len % args.chunk:
        raise SystemExit("seq-len must be a multiple of chunk")
    if args.bank_dtype is None:
        args.bank_dtype = "bfloat16" if args.lag == "none" else "float32"
    if args.lag != "none":
        # Checked here, before the immutable run directory exists.
        if args.bank_dtype != "float32":
            raise SystemExit("a lagged core keeps fp32 banks")
        if args.seq_len % args.lag_chunk:
            raise SystemExit("seq-len must be a multiple of lag-chunk")
        if not 1 <= args.lag_chunk <= MAX_LAG_CHUNK:
            raise SystemExit(f"lag-chunk must lie in [1, {MAX_LAG_CHUNK}]")
    for tokens in (args.val_tokens, args.final_val_tokens):
        if tokens % (args.seq_len * args.eval_microbatch):
            raise SystemExit("validation tokens must fill whole eval microbatches")
    return args


def main():
    args = parse_args()
    # The chunk-parallel core's fp32 matmuls are bank math (decay-factored
    # WY products); TF32 would round them to 10 mantissa bits.
    torch.backends.cuda.matmul.allow_tf32 = args.lag == "none"
    torch.backends.cudnn.allow_tf32 = True
    run_dir = args.results_root / "ablation_results" / args.name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; run names are immutable")
    run_dir.mkdir(parents=True)
    source = run_dir / "source"
    shutil.copytree(Path(__file__).parent, source / "cwm", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO / "pretraining/nanogpt_mini/nanogpt_mini_kda_model.py", source)
    shutil.copy2(REPO / "pretraining/hbyte/train.py", source / "hbyte_train.py")

    config = CWMConfig(
        rounds=args.rounds,
        perception="attention" if args.variant == "cwm" else "kda",
        store=args.variant == "cwm",
        read_mask=args.read_mask,
        bank_dtype=args.bank_dtype,
        lag=args.lag,
        lag_chunk=args.lag_chunk,
    )
    model = CWM(config).cuda()
    initialize(model, args.seed)
    optimizers = build_optimizers(model, args)
    objective = Objective(model, args)
    lut = torch.load(BYTE_LUT).cuda().long()
    val_inputs, val_targets = validation_rows(args.final_val_tokens, args.seq_len)
    val_rows = args.val_tokens // args.seq_len
    stream = ContiguousStream(DATA, args.rows_per_step, args.seq_len)
    manifest = {
        "name": args.name,
        "architecture": ARCHITECTURE,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_config": config_dict(config),
        "parameters": parameter_report(model),
        "data": {
            "train_shards": [{"path": str(f), "bytes": f.stat().st_size} for f in stream.files],
            "val_shard": str(sorted(DATA.glob("fineweb_val_*.bin"))[0]),
            "val_tokens": args.val_tokens,
            "final_val_tokens": args.final_val_tokens,
            "gpt2_byte_lut_sha256": file_sha256(BYTE_LUT),
        },
        "git_revision": git_revision(),
        "torch": torch.__version__,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["parameters"]), flush=True)
    log = MetricLog(run_dir, args.results_root / "tb_logs" / args.name)

    def run_eval(step: int, final: bool) -> dict:
        t0 = time.perf_counter()
        rows = val_inputs.size(0) if final else val_rows
        result = evaluate(objective, val_inputs[:rows], val_targets[:rows], lut, args.eval_microbatch)
        if config.read_mask != "lattice":
            # The same weights with every feedback read removed: what the
            # trained feedback channel contributes (not a matched control --
            # the lattice arm is).
            model.feedback_enabled = False
            ablated = evaluate(objective, val_inputs[:val_rows], val_targets[:val_rows], lut, args.eval_microbatch)
            model.feedback_enabled = True
            reference = result if not final else evaluate(
                objective, val_inputs[:val_rows], val_targets[:val_rows], lut, args.eval_microbatch
            )
            result["feedback_ablated_bpb"] = ablated["bpb"]
            result["feedback_contribution_bpb"] = ablated["bpb"] - reference["bpb"]
            for lo in CONTEXT_BINS:
                result[f"feedback_ablated_ctx_{lo}_bpb"] = ablated[f"ctx_{lo}_bpb"]
        result["eval_seconds"] = time.perf_counter() - t0
        result.update(gpu_neighbors())
        log.write("val", step, result)
        print(f"step {step} val_bpb {result['bpb']:.4f}", flush=True)
        return result

    accum = args.rows_per_step // args.microbatch
    train_nll = torch.zeros((), device="cuda")
    train_bytes = torch.zeros((), device="cuda", dtype=torch.long)
    train_tokens = 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    # Squared global gradient norms of the logging window (non-finite values
    # propagate, so one bad step is caught at the next log without a sync).
    grad_sq = torch.zeros((), device="cuda")
    grad_sq_max = torch.zeros((), device="cuda")
    grad_steps = 0
    step_time = train_seconds = 0.0
    window_steps = 0
    val = run_eval(0, final=False)
    for step in range(args.steps):
        t0 = time.perf_counter()
        eta = set_schedule(optimizers, step, args.steps)
        inputs, targets = stream.next_batch()
        for micro in range(accum):
            rows = slice(micro * args.microbatch, (micro + 1) * args.microbatch)
            train_nll += objective.train(inputs[rows], targets[rows])
        train_bytes += lut[targets].sum()
        train_tokens += targets.numel()
        done = step + 1
        if done % 50 == 0 or done == 1:
            log.write("grad", done, component_grad_norms(model))
        norm_sq = torch.stack(torch._foreach_norm([p.grad for p in trainable])).square().sum()
        grad_sq += norm_sq
        grad_sq_max = torch.maximum(grad_sq_max, norm_sq)
        grad_steps += 1
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        if done == 1:
            # The first step compiles: it trains but is not timed.
            torch.cuda.synchronize()
            log.write("compile", done, {"first_step_seconds": time.perf_counter() - t0})
            t0 = time.perf_counter()
        else:
            window_steps += 1
        last = done == args.steps
        log_step = (done % args.log_every == 0 or last) and window_steps > 0
        if log_step:
            torch.cuda.synchronize()
        step_time += time.perf_counter() - t0
        if log_step:
            nats = train_nll.item()
            grad_rms = math.sqrt(grad_sq.item() / grad_steps)
            if not (math.isfinite(nats) and math.isfinite(grad_rms)):
                log.write("stop", done, {"reason": "non-finite train loss or gradient", "loss_nats": nats, "grad_norm": grad_rms})
                raise SystemExit(f"non-finite train loss or gradient by step {done}")
            train_seconds += step_time
            log.write(
                "train",
                done,
                {
                    "loss": nats / train_tokens,
                    "bpb": nats / LN2 / train_bytes.item(),
                    "lr_scale": eta,
                    "step_ms": 1000 * step_time / window_steps,
                    "train_seconds": train_seconds,
                    "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "grad_norm_rms": grad_rms,
                    "grad_norm_max": grad_sq_max.sqrt().item(),
                },
            )
            print(f"step {done} train_loss {nats / train_tokens:.4f} step_ms {1000 * step_time / window_steps:.0f}", flush=True)
            train_nll.zero_(), train_bytes.zero_(), grad_sq.zero_(), grad_sq_max.zero_()
            train_tokens, step_time, window_steps, grad_steps = 0, 0.0, 0, 0
        if done % 50 == 0 or last:
            log.write("channels", done, channel_norms(model))
        if done % args.val_every == 0 or last:
            val = run_eval(done, final=last)
            if not math.isfinite(val["bpb"]):
                raise SystemExit(f"non-finite validation at step {done}")
    if not args.no_save:
        torch.save(
            {
                "model": model.state_dict(),
                "config": config_dict(config),
                "architecture": ARCHITECTURE,
                "tokenizer": {"name": "gpt2", "vocab_size": config.vocab_size, "byte_lut": str(BYTE_LUT)},
                "step": args.steps,
                "manifest": manifest,
            },
            run_dir / "final_model.pt",
        )
    result = {"name": args.name, "final_step": args.steps, "final_val_bpb": val["bpb"], "train_seconds": train_seconds}
    for key in ("feedback_ablated_bpb", "feedback_contribution_bpb"):
        if key in val:
            result[key] = val[key]
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""Train a hierarchical byte model or its GPT-2 control on matched text.

Recipe (from the matched token reference `nanogpt_latest_mathglm_v6_matched_2k_v7`):
Muon 0.025 (per-head for Q/K/V, wd 0.05, momentum 0.85 -> 0.95 over 500
steps) on block matrices; AdamW (0.8, 0.95), wd 0.001 with lookup 0.7, head
0.004, conv 0.004, scalars 0.015; cosine schedule with 1% warmup over
``--schedule-steps``, stopped after ``--steps``. The reference's NextLat
auxiliary is not used here (it cost +0.0225 BPB and 2.2x wall-clock on the v7
factorial), so the token control in this trainer is the clean comparand.

Outputs: ``ablation_results/<name>/{manifest.json,metrics.jsonl,source/,
final_model.pt}`` and TensorBoard events in ``tb_logs/<name>``.

Run through mlq, e.g.:

    mlq submit --name hbyte_static4 --cwd "$PWD" --max-parallel-runs 1 -- \
      .venv/bin/python -m pretraining.hbyte.train --name hbyte_static4_1k
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

# NextLat's level sizes change every step; without expandable segments the
# caching allocator fragments and stalls (set before CUDA init).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from pretraining.hbyte import data as hdata
from pretraining.hbyte.model import (
    BYTE_EOT,
    ByteLM,
    CausalSelfAttention,
    HashConfig,
    KimiDeltaAttention,
    LocalAttention,
    LocalConfig,
    ModelConfig,
    TokenLM,
    TrunkConfig,
    build_model,
    capped_ends,
    flops_per_atom,
    initialize,
    lookup_tables,
    parameter_report,
    spacelike_ends,
)

LN2 = math.log(2)
REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Muon (quintic Newton-Schulz, as in the token trainer)
# ---------------------------------------------------------------------------


def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if transposed else X


@torch.compile
def muon_update(grad: Tensor, momentum: Tensor, mu: Tensor) -> Tensor:
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu)
    update = zeropower_via_newtonschulz5(update)
    return update * max(1, grad.size(-2) / grad.size(-1)) ** 0.5


class Muon(torch.optim.Optimizer):
    """Single-process Muon; ``heads`` > 1 orthogonalizes each head's rows."""

    def __init__(self, params_with_heads, lr: float, weight_decay: float, mu: float):
        self.heads = {id(p): h for p, h in params_with_heads}
        params = sorted((p for p, _ in params_with_heads), key=lambda p: p.size(), reverse=True)
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, mu=mu))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if not state:
                    state["momentum"] = torch.zeros_like(p)
                    state["mu"] = torch.empty((), device=p.device)
                state["mu"].fill_(group["mu"])
                heads = self.heads[id(p)]
                shape = (heads, p.size(0) // heads, p.size(1))
                update = muon_update(p.grad.view(shape), state["momentum"].view(shape), state["mu"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape_as(p), alpha=-group["lr"])


def build_optimizers(model: nn.Module, args) -> list[torch.optim.Optimizer]:
    if isinstance(model, TokenLM):
        lookup = [model.embed.weight]
        head = [model.head.weight]
    else:
        lookup = [t.weight for t in lookup_tables(model)]
        head = [model.decoder.head.weight]
        for extra in ("trunk_head", "prefix_head"):
            if getattr(model.decoder, extra, None) is not None:
                head.append(getattr(model.decoder, extra).weight)
        if model.patcher is not None:  # entropy head, or router projections
            head += [p for p in model.patcher.parameters() if p.ndim == 2]
    decay, conv, per_head = [], [], []
    for module in model.modules():
        if isinstance(module, KimiDeltaAttention):
            decay += [module.A_log, module.dt_bias]
            conv += [c.weight for c in (module.q_conv1d, module.k_conv1d, module.v_conv1d)]
            per_head += [(l.weight, module.num_heads) for l in (module.q_proj, module.k_proj, module.v_proj)]
        elif isinstance(module, (CausalSelfAttention, LocalAttention)):
            per_head += [(l.weight, module.num_heads) for l in (module.q, module.k, module.v)]
    special = {id(p) for p in lookup + head + decay + conv} | {id(p) for p, _ in per_head}
    scalars = [p for p in model.parameters() if p.ndim < 2 and id(p) not in special]
    matrices = [(p, 1) for p in model.parameters() if p.ndim == 2 and id(p) not in special]
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
    if len(owned) != len({id(p) for p in owned}) or {id(p) for p in owned} != {id(p) for p in model.parameters()}:
        raise RuntimeError("every parameter must belong to exactly one optimizer group")
    for opt in (adam, muon):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return [adam, muon]


def set_schedule(optimizers, step: int, schedule_steps: int, warmup_fraction=0.01, momentum_warmup=500):
    warmup = max(1, round(schedule_steps * warmup_fraction))
    if step < warmup:
        eta = (step + 1) / warmup
    else:
        eta = 0.5 * (1 + math.cos(math.pi * (step - warmup) / (schedule_steps - warmup)))
    mu = 0.85 + min(step / momentum_warmup, 1.0) * (0.95 - 0.85)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta
            if "mu" in group:
                group["mu"] = mu
    return eta


# ---------------------------------------------------------------------------
# Evaluation and diagnostics
# ---------------------------------------------------------------------------

BYTE_CLASSES = ("space", "word_initial", "word_internal", "digit", "punct", "non_ascii", "eot")
CONTEXT_BINS = (0, 64, 512, 2048)
# Gate-score bins for calibration against the model's own loss: the entropy
# patcher's predicted entropy (bits), or the learned router's end probability.
GATE_BINS = {"entropy": (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0), "learned": tuple(i / 10 for i in range(10))}
SAMPLE_BYTES = 600


def byte_classes(inputs: Tensor, targets: Tensor) -> Tensor:
    y, prev = targets, inputs
    alpha = lambda t: ((t >= 65) & (t <= 90)) | ((t >= 97) & (t <= 122))
    digit = lambda t: (t >= 48) & (t <= 57)
    space = (y == 32) | (y == 9) | (y == 10) | (y == 13)
    cls = torch.full_like(y, 4)  # punctuation / other ASCII
    cls = torch.where(alpha(y) & ~(alpha(prev) | digit(prev)), 1, cls)
    cls = torch.where(alpha(y) & (alpha(prev) | digit(prev)), 2, cls)
    cls = torch.where(digit(y), 3, cls)
    cls = torch.where(space, 0, cls)
    cls = torch.where((y >= 128) & (y < 256), 5, cls)
    return torch.where(y == BYTE_EOT, 6, cls)


def length_bins(cap: int) -> tuple[int, ...]:
    """Lower edges for offsets (0 .. cap-1) and patch lengths (1 .. cap):
    exact up to 4, then octaves."""
    edges = [0, 1, 2, 3, 4]
    while edges[-1] * 2 <= cap:
        edges.append(edges[-1] * 2)
    return tuple(e for e in edges if e <= cap)


def bin_names(edges: tuple[int, ...], top: int) -> list[str]:
    """``3`` for a one-value bin, ``8-15`` for a range (``top`` inclusive)."""
    his = [e - 1 for e in edges[1:]] + [top]
    return [str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in zip(edges, his)]


def patch_lengths(ends: Tensor) -> Tensor:
    """Length of the patch each position is decoded in: position ``t``
    (target ``t+1``) reads the patch that ended at ``last_end(t)``, and the
    bytes it predicts form the next patch, of length ``next_end - last_end``
    where ``next_end`` is the first end strictly after ``last_end``. Zero
    before a row's first end and after its last (incomplete patches)."""
    T = ends.size(1)
    t = torch.arange(T, device=ends.device)
    last_end = torch.where(ends, t, -1).cummax(1).values
    upcoming = torch.where(ends, t, T).flip(1).cummin(1).values.flip(1)
    after = torch.cat([upcoming[:, 1:], torch.full_like(upcoming[:, :1], T)], 1)
    next_end = torch.where(ends, after, upcoming)
    return torch.where((last_end >= 0) & (next_end < T), next_end - last_end, 0)


def render_patches(inputs: Tensor, ends: Tensor, valid: Tensor) -> str:
    """Scored bytes with ``|`` after every patch end."""
    chars = []
    for b, e in zip(inputs[valid].tolist(), ends[valid].tolist()):
        chars.append("\u23ce" if b == 10 else "\u2403" if b == BYTE_EOT else chr(b) if 32 <= b < 127 else "\u00b7")
        if e:
            chars.append("|")
    return "".join(chars)


def matched_bernoulli_rate(target: float, cap: int, valid: Tensor, seed: int = 0) -> float:
    """Bernoulli end probability whose realized rate, after the cap's forced
    ends, is ``target`` on these positions (bisection)."""
    gen = torch.Generator(device=valid.device).manual_seed(seed)
    u = torch.rand(valid.shape, device=valid.device, generator=gen)
    lo, hi = 0.0, target
    for _ in range(30):
        mid = (lo + hi) / 2
        realized = float((capped_ends(u < mid, cap) & valid).sum() / valid.sum())
        lo, hi = (mid, hi) if realized < target else (lo, mid)
    return (lo + hi) / 2


@torch.no_grad()
def evaluate(model, eval_set: hdata.EvalSet, microbatch: int, diagnostics=False, **kwargs) -> dict:
    """Exact codelength over ``eval_set``; optional byte-level breakdowns
    (byte class, context position, offset within patch, length of the
    patch, gate calibration, boundary samples). ``model`` is a ByteLM (its
    parts compiled) or a compiled TokenLM; ``kwargs`` go to its forward."""
    byte = isinstance(model, ByteLM)
    gated = byte and model.patcher is not None
    spaced = byte and model.config.patcher == "space"
    gate_kind = model.config.patcher if gated else "entropy"
    gate_bins = GATE_BINS[gate_kind]
    gate_name = {"entropy": "gate_entropy", "learned": "gate_p"}[gate_kind]
    cap = model.config.patch_cap if byte else 1
    edges = length_bins(cap)  # patch lengths 1 .. cap
    off_edges = tuple(e for e in edges if e < cap)  # offsets 0 .. cap-1
    patches = torch.zeros((), device="cuda", dtype=torch.float64)
    aux_total = torch.zeros((), device="cuda", dtype=torch.float64)
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    forced = torch.zeros((), device="cuda", dtype=torch.float64)
    dropped = torch.zeros((), device="cuda", dtype=torch.float64)

    def sums(n: int) -> tuple[Tensor, Tensor]:
        return torch.zeros(n, device="cuda", dtype=torch.float64), torch.zeros(n, device="cuda", dtype=torch.float64)

    cls_nll, cls_n = sums(len(BYTE_CLASSES))
    ctx_nll, ctx_n = sums(len(CONTEXT_BINS))
    off_nll, off_n = sums(len(off_edges))
    len_nll, len_n = sums(len(edges))
    ent_nll, ent_n = sums(len(gate_bins))
    ent_bits = torch.zeros(len(gate_bins), device="cuda", dtype=torch.float64)
    moments = torch.zeros(5, device="cuda", dtype=torch.float64)  # sum x, y, xx, yy, xy
    ctx_edges = torch.tensor(CONTEXT_BINS, device="cuda")
    len_edges = torch.tensor(edges, device="cuda")
    off_edges_t = torch.tensor(off_edges, device="cuda")
    bits_edges = torch.tensor(gate_bins, device="cuda", dtype=torch.float64)
    sample_rows = {0, eval_set.rows // 3, 2 * eval_set.rows // 3}
    samples = []
    for start in range(0, eval_set.rows, microbatch):
        inputs = eval_set.inputs[start : start + microbatch].cuda(non_blocking=True)
        targets = eval_set.targets[start : start + microbatch].cuda(non_blocking=True)
        rows = inputs.size(0)
        if rows < microbatch:  # keep compiled shapes static
            pad = microbatch - rows
            inputs = torch.cat([inputs, inputs[:1].expand(pad, -1)])
            targets = torch.cat([targets, torch.full_like(targets[:1], -100).expand(pad, -1)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nll, aux = model(inputs, targets, **kwargs)
        valid = targets >= 0
        nll = nll.double() * valid
        total += nll.sum()
        if byte:
            ends = model.last_layout.offset == 0
            # Offset 0 marks a patch end: this counts patches over scored bytes.
            patches += (ends & valid).sum()
            if aux is not None:
                aux_total += (aux.double() * valid).sum()
        if not diagnostics:
            continue
        v = nll[valid]
        cls = byte_classes(inputs, targets)[valid]
        cls_nll += torch.bincount(cls, weights=v, minlength=len(BYTE_CLASSES))
        cls_n += torch.bincount(cls, minlength=len(BYTE_CLASSES)).double()
        pos = torch.arange(1, inputs.size(1) + 1, device="cuda").expand_as(targets)[valid]
        ctx = torch.bucketize(pos, ctx_edges, right=True) - 1
        ctx_nll += torch.bincount(ctx, weights=v, minlength=len(CONTEXT_BINS))
        ctx_n += torch.bincount(ctx, minlength=len(CONTEXT_BINS)).double()
        if not byte:
            continue
        off = torch.bucketize(model.last_layout.offset[valid], off_edges_t, right=True) - 1
        off_nll += torch.bincount(off, weights=v, minlength=len(off_edges))
        off_n += torch.bincount(off, minlength=len(off_edges)).double()
        length = patch_lengths(ends)
        inside = valid & (length > 0)
        lb = torch.bucketize(length[inside], len_edges, right=True) - 1
        len_nll += torch.bincount(lb, weights=nll[inside], minlength=len(edges))
        len_n += torch.bincount(lb, minlength=len(edges)).double()
        # Ends the patcher did not choose were forced by the cap; chosen ends
        # past the row's chunk budget were dropped. Counted over scored bytes.
        if gated:
            natural = model.patcher.natural_ends(model.last_gate)
        elif spaced:
            natural = spacelike_ends(inputs)
        if gated or spaced:
            forced += (ends & ~natural & valid).sum()
            dropped += (natural & ~ends & valid).sum()
        if gated:
            x = model.last_gate[valid].double()  # gate score: entropy (bits) or p
            if gate_kind == "entropy":
                x = x / LN2
            y = v / LN2  # realized codelength, bits
            eb = torch.bucketize(x, bits_edges, right=True) - 1
            ent_nll += torch.bincount(eb, weights=v, minlength=len(gate_bins))
            ent_n += torch.bincount(eb, minlength=len(gate_bins)).double()
            ent_bits += torch.bincount(eb, weights=x, minlength=len(gate_bins))
            moments += torch.stack([x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()])
        for r in range(rows):
            if start + r in sample_rows:
                idx = valid[r].nonzero().flatten()
                keep = torch.zeros_like(valid[r])
                keep[idx[len(idx) // 4 : len(idx) // 4 + SAMPLE_BYTES]] = True
                samples.append(render_patches(inputs[r].cpu(), ends[r].cpu(), keep.cpu()))
    out = {"bpb": total.item() / LN2 / eval_set.scored_bytes}
    if byte:
        out["bytes_per_patch"] = eval_set.scored_bytes / patches.item()
        if gated and gate_kind == "entropy":
            out["aux_bpb"] = aux_total.item() / LN2 / eval_set.scored_bytes
    if not diagnostics:
        return out

    def ratio(num: Tensor, den: Tensor, i: int) -> float | None:
        return (num[i] / den[i]).item() / LN2 if den[i] > 0 else None

    for i, name in enumerate(BYTE_CLASSES):
        out[f"class_{name}_bpb"] = ratio(cls_nll, cls_n, i)
        out[f"class_{name}_share"] = (cls_nll[i] / total).item()  # of codelength
    for i, lo in enumerate(CONTEXT_BINS):
        out[f"ctx_{lo}_bpb"] = ratio(ctx_nll, ctx_n, i)
    if not byte:
        return out
    for i, name in enumerate(bin_names(off_edges, cap - 1)):
        if off_n[i] > 0:
            out[f"offset_{name}_bpb"] = ratio(off_nll, off_n, i)
            out[f"offset_{name}_share"] = (off_n[i] / off_n.sum()).item()  # of bytes
    for i, name in enumerate(bin_names(edges, cap)):
        if len_n[i] > 0:
            out[f"patchlen_{name}_bpb"] = ratio(len_nll, len_n, i)
            out[f"patchlen_{name}_share"] = (len_n[i] / len_n.sum()).item()  # of bytes
    if gated or spaced:
        out["forced_end_fraction"] = (forced / patches).item()
        out["dropped_end_fraction"] = (dropped / (patches - forced + dropped)).item()  # of chosen ends
    if gated:
        n = ent_n.sum()
        sx, sy, sxx, syy, sxy = (moments / n).tolist()
        var = (sxx - sx * sx) * (syy - sy * sy)
        out[f"{gate_name}_nll_corr"] = (sxy - sx * sy) / math.sqrt(var) if var > 0 else None
        for i, lo in enumerate(gate_bins):
            if ent_n[i] > 0:
                out[f"{gate_name}_{lo:g}_bpb"] = ratio(ent_nll, ent_n, i)
                out[f"{gate_name}_{lo:g}_mean"] = (ent_bits[i] / ent_n[i]).item()
                out[f"{gate_name}_{lo:g}_share"] = (ent_n[i] / n).item()
    out["samples"] = "\n\n".join(samples)
    return out


SWEEP = (0.6, 0.8, 0.9, 1.1, 1.25, 1.5)


def boundary_counterfactuals(model: ByteLM, eval_set: hdata.EvalSet, microbatch: int, gate: dict, sweep: bool) -> dict:
    """Is the gate placing ends well? The same weights scored with random
    ends at the gate's realized rate and with static ends at the nearest
    stride (both causal: independent of the data), and optionally with the
    gate's threshold scaled (BPB against bytes/patch around the operating
    point). Only run at diagnostic evals."""
    cap = model.config.patch_cap
    rate = matched_bernoulli_rate(1 / gate["bytes_per_patch"], cap, (eval_set.targets >= 0).cuda())
    gen = torch.Generator(device="cuda")

    def random_ends(entropy: Tensor) -> Tensor:
        return capped_ends(torch.rand(entropy.shape, device=entropy.device, generator=gen) < rate, cap)

    stride = max(1, min(cap, round(gate["bytes_per_patch"])))

    def static_ends(entropy: Tensor) -> Tensor:
        return ((torch.arange(entropy.size(1), device=entropy.device) + 1) % stride == 0).expand_as(entropy)

    out = {"static_stride": stride}
    gen.manual_seed(0)
    for name, fn in (("random", random_ends), ("static", static_ends)):
        r = evaluate(model, eval_set, microbatch, boundaries=fn)
        out[f"{name}_bpb"] = r["bpb"]
        out[f"{name}_bytes_per_patch"] = r["bytes_per_patch"]
        out[f"{name}_minus_gate_bpb"] = r["bpb"] - gate["bpb"]
    if sweep:
        threshold = model.patcher.threshold.clone()
        try:
            for m in SWEEP:
                model.patcher.threshold.copy_(threshold * m)
                r = evaluate(model, eval_set, microbatch)
                out[f"sweep_{m:g}_bpb"] = r["bpb"]
                out[f"sweep_{m:g}_bytes_per_patch"] = r["bytes_per_patch"]
        finally:
            model.patcher.threshold.copy_(threshold)
    return out


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class MetricLog:
    """Canonical JSONL plus TensorBoard: ``<kind>/<key>`` against step, and
    the repository's ``time/`` charts (train and val BPB against training
    seconds, evaluation excluded) so runs of different speed line up."""

    def __init__(self, run_dir: Path, tb_dir: Path):
        self.path = run_dir / "metrics.jsonl"
        self.writer = SummaryWriter(str(tb_dir))
        self.train_seconds = 0.0

    def write(self, kind: str, step: int, values: dict) -> None:
        if kind == "train":
            self.train_seconds = values["train_seconds"]
        if kind in ("train", "val"):
            values = {**values, "train_seconds": self.train_seconds}
            self.writer.add_scalar(f"time/{kind}_bpb", values["bpb"], round(self.train_seconds))
        record = {"type": kind, "step": step, **values}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        for key, value in values.items():
            if isinstance(value, str):
                self.writer.add_text(f"{kind}/{key}", "    " + value.replace("\n", "\n    "), step)
            elif isinstance(value, (int, float)) and math.isfinite(value):
                self.writer.add_scalar(f"{kind}/{key}", value, step)
        self.writer.flush()


def gpu_neighbors() -> dict:
    """Other processes holding memory on the GPU, so a contended timing is
    visible in the record (a co-resident game slowed every part by 25-30%)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"gpu_neighbors": "unavailable"}
    others = []
    for line in out.strip().splitlines():
        # Process names are command lines and may contain commas; memory may
        # be "[N/A]" on some platforms.
        pid, rest = line.split(",", 1)
        name, mib = (field.strip() for field in rest.rsplit(",", 1))
        if pid.strip() != str(os.getpid()):
            others.append((Path(name.split()[0]).name if name else "?", int(mib) if mib.isdigit() else 0))
    return {
        "gpu_neighbor_mib": sum(mib for _, mib in others),
        "gpu_neighbors": ", ".join(f"{name}:{mib}" for name, mib in sorted(others, key=lambda o: -o[1])),
    }


def component_grad_norms(model: nn.Module) -> dict:
    groups: dict[str, list[Tensor]] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            groups.setdefault(name.split(".")[0], []).append(p.grad.float().norm())
    return {f"grad_norm_{k}": torch.stack(v).norm().item() for k, v in groups.items()}


def component_timing(model: nn.Module, inputs: Tensor, targets: Tensor, repeats: int = 5) -> dict:
    """Forward+backward milliseconds of each byte-model component (the
    compiled parts the training step runs) on a real microbatch."""
    if isinstance(model, TokenLM):
        return {}
    mask, enc_mask = model.block_masks[inputs.size(1)], model.encoder_masks[inputs.size(1)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(inputs, targets)
        layout = model.last_layout
        trunk_extra, confidence = model.last_trunk_args, model.last_confidence
        h = model.part("encoder")(inputs, enc_mask)
        ends = model.gather_ends(h, layout)
        z = model.part("trunk")(ends, *trunk_extra)

    def timed(fn) -> float:
        for _ in range(2):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats

    def enc_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.part("encoder")(inputs, enc_mask).float().square().mean().backward()

    def trunk_step():
        x = ends.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.part("trunk")(x, *trunk_extra).float().square().mean().backward()

    def dec_step():
        hh = h.clone().requires_grad_()
        zz = z.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.decode(hh, zz, layout, targets, mask, confidence=confidence).mean().backward()

    out = {"ms_encoder": timed(enc_step), "ms_trunk": timed(trunk_step), "ms_decoder": timed(dec_step)}
    if model.patcher is not None:

        def patch_step():
            hh = h.clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if model.config.patcher == "entropy":
                    model.part("patcher")(hh, targets)[1].mean().backward()
                else:
                    model.part("patcher")(hh).mean().backward()

        out["ms_patcher"] = timed(patch_step)
    model.zero_grad(set_to_none=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--atom", choices=("byte", "gpt2"), default="byte")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--schedule-steps", type=int, default=2000)
    p.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="train seconds (eval excluded): replaces --steps, and the schedule follows elapsed/budget",
    )
    p.add_argument("--warmup-fraction", type=float, default=0.01, help="linear LR warmup, as a fraction of the schedule")
    p.add_argument("--seq-len", type=int, default=None, help="atoms per row (byte 8192, gpt2 2048)")
    p.add_argument("--rows-per-step", type=int, default=None, help="byte 216 (~524,288 tokens of text), gpt2 256")
    p.add_argument("--microbatch", type=int, default=None)
    p.add_argument("--eval-microbatch", type=int, default=None)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--diag-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--patch-size", type=int, default=4, help="static patcher stride")
    p.add_argument("--patcher", choices=("static", "space", "entropy", "learned"), default="static")
    p.add_argument("--ratio-weight", type=float, default=0.03, help="learned patcher: chunk-rate loss weight")
    p.add_argument(
        "--threshold-mode",
        choices=("rate", "mean", "fixed"),
        default="rate",
        help="gate threshold: rate (held at --target-patch), mean (entropy), fixed 0.5 (learned, H-Net)",
    )
    p.add_argument("--target-patch", type=float, default=4.0, help="gated patchers: mean bytes per patch")
    p.add_argument("--max-patch", type=int, default=8)
    p.add_argument("--chunk-slack", type=float, default=1.25, help="gated patchers: trunk slots per row / nominal chunks")
    p.add_argument("--aux-weight", type=float, default=1.0, help="weight of the patcher's local next-byte CE")
    p.add_argument("--aux-detach", action="store_true", help="train the patcher head on detached encoder states")
    p.add_argument("--decoder", choices=("attention", "nextlat"), default="attention")
    p.add_argument("--nextlat-proj-factor", type=float, default=1.6, help="NextLat step MLP width / (2 x local dim)")
    p.add_argument("--nextlat-local-reset", action="store_true", help="patch-end state also sees the end byte's encoder state")
    p.add_argument("--patch-pool", choices=("end", "attn"), default="end", help="trunk input per patch")
    p.add_argument(
        "--prefix-bits", type=int, default=0, help="block-anchored prefix memory: log2 hashed rows per table (0 = off)"
    )
    p.add_argument("--block-ngram", type=int, default=1, help="prefix memory: trunk reads block n-grams up to this order")
    p.add_argument("--memory-gate", action="store_true", help="prefix memory: Engram context gate on memory reads")
    p.add_argument(
        "--prefix-context", action="store_true", help="prefix memory: decoder also reads (previous block, prefix) ids"
    )
    p.add_argument(
        "--prefix-local", action="store_true", help="prefix memory: readout modulation also sees the decoder state"
    )
    p.add_argument(
        "--trunk-logits", action="store_true", help="attention decoder: zero-init per-offset logits from the routed trunk state"
    )
    p.add_argument("--local-dim", type=int, default=512)
    p.add_argument("--encoder-layers", type=int, default=1)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--local-window", type=int, default=512)
    p.add_argument("--encoder-window", type=int, default=0, help="encoder attention window (0 = --local-window)")
    p.add_argument("--encoder-mixers", default="", help="per encoder layer: a (window attention) or k (KDA)")
    p.add_argument("--decoder-mixers", default="", help="per decoder layer: a (window attention) or k (KDA)")
    p.add_argument("--local-kda-heads", type=int, default=2)
    p.add_argument("--ngram-orders", default="3,4,5,6,7,8")
    p.add_argument("--ngram-buckets", type=int, default=65_536)
    p.add_argument("--embed-lr", type=float, default=0.7)
    p.add_argument("--head-lr", type=float, default=0.004)
    p.add_argument("--scalar-lr", type=float, default=0.015)
    p.add_argument("--conv-lr", type=float, default=0.004)
    p.add_argument("--muon-lr", type=float, default=0.025)
    p.add_argument("--adam-wd", type=float, default=0.001)
    p.add_argument("--muon-wd", type=float, default=0.05)
    p.add_argument("--cull", action="append", default=[], help="STEP:MAX_VAL_BPB; stop when exceeded")
    p.add_argument(
        "--target-bpb",
        type=float,
        default=None,
        help="speedrun score: stop at the first val bpb at or below this and log the train seconds to reach it",
    )
    p.add_argument("--target-margin", type=float, default=0.03, help="validate densely once val is within this of the target")
    p.add_argument("--target-val-every", type=int, default=20)
    p.add_argument("--time-components", action="store_true")
    p.add_argument("--results-root", type=Path, default=REPO, help="parent of ablation_results/ and tb_logs/")
    args = p.parse_args()
    byte = args.atom == "byte"
    args.seq_len = args.seq_len or (8192 if byte else 2048)
    args.rows_per_step = args.rows_per_step or (216 if byte else 256)
    args.microbatch = args.microbatch or (12 if byte else 16)
    args.eval_microbatch = args.eval_microbatch or args.microbatch
    if args.rows_per_step % args.microbatch:
        raise SystemExit("rows-per-step must be a multiple of microbatch")
    if not 0 < args.steps <= args.schedule_steps:
        raise SystemExit("need 0 < steps <= schedule-steps")
    args.cull = [tuple(float(v) for v in c.split(":")) for c in args.cull]
    if args.target_bpb is not None and (args.val_every % args.log_every or args.target_val_every % args.log_every):
        # Validation then lands on a synchronized log step, so its train
        # seconds are exact rather than the previous window's.
        raise SystemExit("--target-bpb needs val cadences that are multiples of --log-every")
    return args


def model_config(args) -> ModelConfig:
    return ModelConfig(
        atom=args.atom,
        seq_len=args.seq_len,
        trunk=TrunkConfig(),
        local=LocalConfig(
            dim=args.local_dim,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            window=args.local_window,
            encoder_window=args.encoder_window,
            encoder_mixers=args.encoder_mixers,
            decoder_mixers=args.decoder_mixers,
            kda_heads=args.local_kda_heads,
        ),
        hashing=HashConfig(
            orders=tuple(int(o) for o in args.ngram_orders.split(",") if o),
            buckets=args.ngram_buckets,
        ),
        patch_size=args.patch_size,
        patcher=args.patcher,
        threshold_mode=args.threshold_mode,
        target_patch=args.target_patch,
        max_patch=args.max_patch,
        chunk_slack=args.chunk_slack,
        aux_weight=args.aux_weight,
        ratio_weight=args.ratio_weight,
        aux_detach=args.aux_detach,
        decoder=args.decoder,
        nextlat_proj_factor=args.nextlat_proj_factor,
        nextlat_local_reset=args.nextlat_local_reset,
        patch_pool=args.patch_pool,
        trunk_logits=args.trunk_logits,
        prefix_bits=args.prefix_bits,
        block_ngram=args.block_ngram,
        prefix_context=args.prefix_context,
        memory_gate=args.memory_gate,
        prefix_local=args.prefix_local,
    )


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = args.results_root / "ablation_results" / args.name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; run names are immutable")
    run_dir.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent, run_dir / "source", ignore=shutil.ignore_patterns("__pycache__"))

    config = model_config(args)
    model = build_model(config).cuda()
    initialize(model, args.seed)
    val_set, val_info = hdata.reference_validation(args.atom, args.seq_len)
    extra_sets = []
    if args.atom == "byte":
        extra_sets.append(hdata.reference_validation("byte", args.seq_len, aligned=False)[0])
        extra_sets += hdata.domain_validation(args.seq_len)
    model.prepare("cuda", lengths=(val_set.inputs.size(1),))
    optimizers = build_optimizers(model, args)
    if args.atom == "byte":
        # Every part specializes per row length (training and the longer
        # token-aligned eval rows); the trunk length is the chunk budget.
        model.compile_parts()
        compiled = model
    else:
        compiled = torch.compile(model, dynamic=False)
    byte_lut = torch.load(hdata.GPT2_BYTE_LUT).cuda().long() if args.atom == "gpt2" else None
    stream = hdata.ContiguousStream(
        hdata.GPT2_DATA if args.atom == "gpt2" else hdata.BYTE_DATA, args.rows_per_step, args.seq_len
    )
    manifest = {
        "name": args.name,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_config": dataclasses.asdict(config),
        "parameters": parameter_report(model),
        "train_flops_per_atom": flops_per_atom(model, config),
        "validation": val_info,
        "data": hdata.data_identity(args.atom),
        "git_revision": git_revision(),
        "torch": torch.__version__,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["parameters"]), json.dumps(manifest["train_flops_per_atom"]), flush=True)
    log = MetricLog(run_dir, args.results_root / "tb_logs" / args.name)

    def run_eval(step: int, diagnostics: bool, final: bool = False, kind: str = "val") -> float:
        compiled.eval()
        t0 = time.perf_counter()
        byte = args.atom == "byte"
        result = evaluate(compiled, val_set, args.eval_microbatch, diagnostics and byte)
        if diagnostics and byte:
            # A knock-out the decoder never trained with: an upper bound on
            # what the trunk contributes, not a matched ablation.
            result["global_ablated_bpb"] = evaluate(compiled, val_set, args.eval_microbatch, ablate_global=True)["bpb"]
            result["global_contribution_bpb"] = result["global_ablated_bpb"] - result["bpb"]
            for extra in extra_sets:
                result[f"{extra.name}_bpb"] = evaluate(compiled, extra, args.eval_microbatch)["bpb"]
            if model.patcher is not None:
                result.update(boundary_counterfactuals(model, val_set, args.eval_microbatch, result, sweep=final))
        result["eval_seconds"] = time.perf_counter() - t0
        result.update(gpu_neighbors())
        compiled.train()
        log.write(kind, step, result)
        print(f"step {step} {kind}_bpb {result['bpb']:.4f}", flush=True)
        return result["bpb"]

    accum = args.rows_per_step // args.microbatch
    model_has_patcher = getattr(model, "patcher", None) is not None
    train_nll = torch.zeros((), device="cuda")
    train_aux = torch.zeros((), device="cuda")
    patch_stats: dict[str, Tensor] = {}
    train_bytes = torch.zeros((), device="cuda")
    train_atoms = 0
    step_time = train_seconds = 0.0
    window_steps = 0
    val_bpb = run_eval(0, diagnostics=False)
    previous_val = (0, 0.0, val_bpb)
    target = args.target_bpb
    if target is not None and val_bpb <= target:
        raise SystemExit(f"target {target} is at or above the initial val bpb {val_bpb:.4f}")
    target_record: dict = {}
    budget = args.time_budget
    step = 0
    while True:
        t0 = time.perf_counter()
        if budget is None:
            eta = set_schedule(optimizers, step, args.schedule_steps, args.warmup_fraction)
        else:
            # The --schedule-steps schedule indexed by the elapsed fraction of
            # the budget, so every budgeted run ends fully annealed.
            elapsed = (train_seconds + step_time) / budget
            virtual = min(int(elapsed * args.schedule_steps), args.schedule_steps - 1)
            eta = set_schedule(optimizers, virtual, args.schedule_steps, args.warmup_fraction)
        inputs, targets = stream.next_batch()
        for micro in range(accum):
            rows = slice(micro * args.microbatch, (micro + 1) * args.microbatch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nll, aux = compiled(inputs[rows], targets[rows])
            # Summed CE, as in the reference trainer.
            loss = nll.sum()
            if aux is not None:
                loss = loss + model.aux_scale * aux.sum()
                train_aux += aux.detach().sum()
            loss.backward()
            train_nll += nll.detach().sum()
            for key, value in getattr(model, "last_stats", {}).items():
                patch_stats[key] = patch_stats.get(key, 0) + value / accum
        train_bytes += byte_lut[targets].sum() if byte_lut is not None else targets.numel()
        train_atoms += targets.numel()
        done = step + 1
        if done % 50 == 0:
            log.write("grad", done, component_grad_norms(model))
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        # Without a budget, the step count ends the run. With one, the
        # unsynchronized wall time (it lags the GPU by at most the queued
        # work) triggers the stop, and the synchronized final window counts.
        if done == 1:
            # The first step compiles every part: it trains, but is not timed.
            torch.cuda.synchronize()
            log.write("compile", done, {"first_step_seconds": time.perf_counter() - t0})
            t0 = time.perf_counter()
        else:
            window_steps += 1
        last = done == args.steps if budget is None else train_seconds + step_time + time.perf_counter() - t0 >= budget
        near = target is not None and val_bpb <= target + args.target_margin
        val_due = done % (args.target_val_every if near else args.val_every) == 0 or last
        log_step = (done % args.log_every == 0 or last) and window_steps > 0
        if log_step:
            torch.cuda.synchronize()
        step_time += time.perf_counter() - t0
        if log_step:
            nats = train_nll.item()
            log.write(
                "train",
                done,
                {
                    "loss": nats / train_atoms,
                    "bpb": nats / LN2 / train_bytes.item(),
                    "lr_scale": eta,
                    "step_ms": 1000 * step_time / window_steps,
                    "steps_per_second": window_steps / step_time,
                    "train_seconds": (train_seconds := train_seconds + step_time),
                    "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                    **({"aux_bpb": train_aux.item() / LN2 / train_bytes.item()} if config.patcher == "entropy" else {}),
                    # Stats of the last step in the window.
                    **{k: float(v) for k, v in patch_stats.items()},
                },
            )
            train_nll.zero_(), train_bytes.zero_(), train_aux.zero_()
            train_atoms, step_time, window_steps = 0, 0.0, 0
        if step == 20 and args.time_components:
            log.write("timing", step, component_timing(model, inputs[: args.microbatch], targets[: args.microbatch]))
        patch_stats = {}
        step = done
        if val_due:
            # In a target run the diagnostics run once, at the end.
            val_bpb = run_eval(done, diagnostics=last or (done % args.diag_every == 0 and target is None), final=last)
            if target is not None and val_bpb <= target:
                # Linear in bpb between the bracketing evaluations.
                step0, seconds0, bpb0 = previous_val
                frac = (bpb0 - target) / (bpb0 - val_bpb) if bpb0 > val_bpb else 1.0
                target_record = {
                    "target_bpb": target,
                    "val_bpb": val_bpb,
                    "train_seconds": train_seconds,
                    "interp_seconds": seconds0 + frac * (train_seconds - seconds0),
                    "interp_step": step0 + frac * (done - step0),
                }
                log.write("target", done, target_record)
                print(f"target {target} reached at step {done}, {train_seconds:.1f} train s", flush=True)
                if not last:  # the last val already ran them
                    run_eval(done, diagnostics=True, final=True, kind="diag")
                break
            previous_val = (done, train_seconds, val_bpb)
            if not math.isfinite(val_bpb) or any(done >= s and val_bpb > m for s, m in args.cull):
                log.write("stop", done, {"reason_val_bpb": val_bpb})
                print(f"culled at step {done}: val_bpb {val_bpb:.4f}", flush=True)
                break
        if last:
            break
    torch.save(
        {"model": model.state_dict(), "config": dataclasses.asdict(config), "step": done, "manifest": manifest},
        run_dir / "final_model.pt",
    )
    result = {"name": args.name, "final_step": done, "final_val_bpb": val_bpb}
    if target is not None:
        result["target_reached"] = bool(target_record)
        result |= {f"target_{k}": v for k, v in target_record.items() if k != "target_bpb"}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

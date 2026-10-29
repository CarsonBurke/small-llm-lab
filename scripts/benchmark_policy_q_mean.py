#!/usr/bin/env python3
"""Investigate exact policy-Q mean readout alternatives without a model.

CPU mode is a cheap algebra/readout check; CUDA mode must use mlq. This
measures the readout tail, not a model training update. The
production functions come from a frozen source file; no model is constructed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch import Tensor


def production_namespace(path):
    source = path.read_text()
    wanted = {
        "rational_softcap", "policy_centered_token_values",
        "SoftcappedTargetLogprobsEntropy",
        "SoftcappedTargetLogprobsEntropyAdvantages",
    }
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if getattr(node, "name", None) in wanted]
    if {node.name for node in tree.body} != wanted:
        raise ValueError("frozen source lacks the production readout definitions")
    namespace = {"torch": torch, "Tensor": Tensor}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def make_readouts(namespace):
    original = namespace["SoftcappedTargetLogprobsEntropyAdvantages"]
    entropy_only = namespace["SoftcappedTargetLogprobsEntropy"]
    softcap = namespace["rational_softcap"]

    class Alternative(torch.autograd.Function):
        @staticmethod
        def forward(ctx, raw, targets, cap, directions, latents, mode):
            capped = softcap(raw.float(), cap)
            logsumexp = capped.logsumexp(-1)
            probabilities = (capped - logsumexp[:, None]).exp()
            entropy = logsumexp - (probabilities * capped).sum(-1)
            with torch.no_grad(), torch.autocast(raw.device.type, enabled=False):
                if mode == "reuse_scores":
                    scores = directions.float() @ latents.float().T
                    taken = scores.gather(-1, targets[:, None]).squeeze(-1)
                    advantage = taken - (probabilities * scores).sum(-1)
                elif mode == "mean_embedding":
                    mean_embedding = probabilities @ latents.float()
                    advantage = (
                        directions.float() * (latents[targets].float() - mean_embedding)
                    ).sum(-1)
                else:
                    raise ValueError(mode)
            target_raw = raw.gather(-1, targets[:, None]).squeeze(-1).float()
            ctx.save_for_backward(raw, targets, logsumexp, entropy)
            ctx.cap = cap
            ctx.mark_non_differentiable(advantage)
            return softcap(target_raw, cap) - logsumexp, entropy, advantage

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_logprob, grad_entropy, _):
            return entropy_only.raw_gradient(ctx, grad_logprob, grad_entropy), None, None, None, None, None

    def existing(raw, targets, directions, latents):
        return original.apply(raw, targets, 15.0, directions, latents)

    def reuse(raw, targets, directions, latents):
        return Alternative.apply(raw, targets, 15.0, directions, latents, "reuse_scores")

    def embedding(raw, targets, directions, latents):
        return Alternative.apply(raw, targets, 15.0, directions, latents, "mean_embedding")

    def no_mean(raw, targets, directions, latents):
        logprob, entropy = entropy_only.apply(raw, targets, 15.0)
        return logprob, entropy, torch.zeros_like(logprob)

    return {"existing": existing, "reuse_scores": reuse, "mean_embedding": embedding, "no_mean": no_mean}


def inputs(rows, vocab, dim, case, device):
    raw = torch.randn(rows, vocab, device=device, dtype=torch.bfloat16)
    targets = torch.randint(vocab, (rows,), device=device)
    dominant = targets.clone()
    if case == "rare_target":
        dominant = (targets + 1) % vocab
    if case in {"peaked", "rare_target"}:
        # Capped dominant logits ~13.4: produces low-entropy policy rows.
        raw.scatter_(1, dominant[:, None], 30.0)
    elif case == "near_deterministic":
        raw.fill_(-100.0)
        raw.scatter_(1, dominant[:, None], 100.0)
    return raw.requires_grad_(), targets


def errors(value, reference):
    delta = value.double() - reference
    scale = reference.square().mean().sqrt().item()
    material = reference.abs() > 1e-6
    return {
        "max_abs": delta.abs().max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "relative_rms": delta.square().mean().sqrt().item() / scale if scale else None,
        "sign_mismatches_above_1e-6": int(((value.sign() != reference.sign()) & material).sum()),
        "reference_rms": scale,
    }


def accuracy(readouts, latents, directions, vocab, rows=8):
    reports = {}
    for case in ("broad", "peaked", "rare_target", "near_deterministic"):
        raw, targets = inputs(rows, vocab, latents.shape[1], case, latents.device)
        u = directions[:rows]
        with torch.no_grad():
            capped = 15 * raw.double() / (raw.double().square() + 225).sqrt()
            probabilities = capped.softmax(-1)
            scores = u.double() @ latents.double().T
            reference = scores.gather(-1, targets[:, None]).squeeze(-1) - (probabilities * scores).sum(-1)
            del scores, probabilities, capped
        observed, gradients = {}, {}
        for name, fn in readouts.items():
            if name == "no_mean":
                continue
            raw.grad = None
            logprob, entropy, advantage = fn(raw, targets, u, latents)
            if advantage.requires_grad:
                raise AssertionError("critic advantage must be detached")
            observed[name] = (logprob.detach(), entropy.detach(), advantage.detach())
            # Production DG coefficient; entropy has zero gradient in this run.
            coefficient = advantage * torch.sigmoid(advantage * -logprob.detach())
            (-coefficient * logprob).mean().backward()
            gradients[name] = raw.grad.detach().clone()
        base = observed["existing"]
        reports[case] = {
            "policy_entropy_mean": base[1].mean().item(),
            "variants": {
                name: {
                    "advantage_vs_float64": errors(outputs[2], reference),
                    "advantage_vs_existing": errors(outputs[2], base[2].double()),
                    "logprob_max_abs_difference": (outputs[0] - base[0]).abs().max().item(),
                    "entropy_max_abs_difference": (outputs[1] - base[1]).abs().max().item(),
                    "dg_raw_logit_gradient_vs_existing": errors(gradients[name], gradients["existing"].double()),
                }
                for name, outputs in observed.items()
            },
        }
        del raw, reference, observed, gradients
    return reports


def measure(fn, args, backward, repeats):
    raw = args[0]

    def iteration():
        raw.grad = None
        outputs = fn(*args)
        if backward:
            # Same upstream derivative for each variant: isolate runtime from
            # numerical differences in the chosen advantage estimator.
            (outputs[0].mean() + 0.0 * outputs[1].mean()).backward()
        return outputs

    for _ in range(3):
        outputs = iteration()
        del outputs
    cuda = raw.is_cuda
    if cuda:
        torch.cuda.synchronize()
    raw.grad = None
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    samples = []
    for _ in range(repeats):
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            started = time.perf_counter()
        outputs = iteration()
        if cuda:
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            samples.append(1000 * (time.perf_counter() - started))
        del outputs
    peak = torch.cuda.max_memory_allocated() - base if cuda else None
    raw.grad = None
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples), "peak_extra_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no device fallback")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    vocab, dim = 50304, 512
    latents = torch.randn(vocab, dim, device=args.device)
    latents = torch.nn.functional.rms_norm(latents, (dim,))
    directions = torch.randn(max(max(args.rows), 8), dim, device=args.device) * (0.003 / dim**0.5)
    readouts = make_readouts(production_namespace(args.source_file))
    report = {
        "schema": "policy_q_mean_readout_benchmark/v1",
        "scope": "synthetic production dimensions; readout tail, excluding logits projection, actor/critic trunks, collection and optimizers; CPU timings do not predict CUDA speed",
        "source_sha256": hashlib.sha256(args.source_file.read_bytes()).hexdigest(),
        "torch": torch.__version__, "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "compiled": args.compile, "threads": args.threads,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "vocab": vocab, "dim": dim, "repeats": args.repeats,
        "accuracy": {}, "timings": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    compiled = {name: torch.compile(fn, fullgraph=True, dynamic=False) for name, fn in readouts.items()} if args.compile else readouts
    report["accuracy"] = accuracy(compiled, latents, directions, vocab)
    save()
    print("accuracy checks complete", flush=True)
    rng = random.Random(1337)
    for rows in args.rows:
        raw, targets = inputs(rows, vocab, dim, "peaked", args.device)
        call_args = (raw, targets, directions[:rows], latents)
        names = list(compiled)
        rng.shuffle(names)
        for name in names:
            for backward in (False, True):
                key = f"rows_{rows}/{name}/{'forward_backward' if backward else 'forward'}"
                report["timings"][key] = measure(compiled[name], call_args, backward, args.repeats)
                save()
                print(key, json.dumps(report["timings"][key]), flush=True)
        del raw, targets, call_args
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()

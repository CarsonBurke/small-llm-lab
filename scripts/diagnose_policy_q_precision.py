"""Tensor-only numerical audit of policy-centered Q; never runs a model.

CUDA invocations must be submitted through mlq. Policies and directions are
controlled diagnostic inputs, not reconstructions of recorded actor prefixes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F


def frozen_helper(source: Path):
    name = "policy_centered_token_values_from_probabilities"
    contents = source.read_text()
    function = next(
        node for node in ast.parse(contents).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"torch": torch, "Tensor": Tensor}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name], hashlib.sha256(contents.encode()).hexdigest()


def summarize(values: Tensor, reference: Tensor) -> dict:
    error = values.double() - reference
    nonzero = reference.abs() > 1e-12
    return {
        "rms": values.double().square().mean().sqrt().item(),
        "max_abs": values.abs().max().item(),
        "error_rms": error.square().mean().sqrt().item(),
        "error_max_abs": error.abs().max().item(),
        "error_over_stop_rms_1e6": error.square().mean().sqrt().item() / 1e-6,
        "error_over_overall_rms_005": error.square().mean().sqrt().item() / .005,
        "sign_flips_nonzero_reference": int(((values.double() * reference < 0) & nonzero).sum()),
        "reference_nonzero_count": int(nonzero.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 128])
    args = parser.parse_args()
    torch.set_num_threads(2)
    helper, source_sha = frozen_helper(args.source)
    # Trusted local training checkpoint, mmap avoids materializing unrelated
    # actor/optimizer tensors. Only the embedding and one readout are used.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    critic = checkpoint["critic"]
    embeddings = critic["trunk.embed.weight"].to(args.device).float()
    gains = critic["trunk.norm1.gains"].to(args.device).float()
    latents = F.rms_norm(F.rms_norm(embeddings, (embeddings.size(-1),), weight=gains),
                         (embeddings.size(-1),))
    direction_weight = critic["q_head.direction.weight"].to(args.device).float()
    latents64 = latents.double()
    vocab, dim = latents.shape
    torch.manual_seed(20261013)
    results = []
    for rows in args.rows:
        targets = torch.randint(vocab, (rows,), device=args.device)
        targets[:min(rows, 4)] = torch.tensor([50256, 50257, 50258, 50259], device=args.device)[:min(rows, 4)]
        # A valid positive ReLU^2 feature input, but not a real prefix feature.
        features = torch.randn(rows, direction_weight.size(1), device=args.device).relu().square() / 3
        torch.set_float32_matmul_precision("highest")
        weight_directions = features @ direction_weight.T
        random_directions = torch.randn(rows, dim, device=args.device) * (.005 / dim**.5)
        for direction_name, directions in (("checkpoint_weight_synthetic_features", weight_directions),
                                            ("random_score_rms_005", random_directions)):
            directions64 = directions.double()
            scores64 = directions64 @ latents64.T
            for rest_mass in (0., 1e-7, 1e-6, 1e-4, .01, .5):
                # Uniform rest mass with chosen token probability 1-rest_mass.
                probabilities = torch.full((rows, vocab), rest_mass / (vocab - 1), device=args.device)
                probabilities.scatter_(1, targets[:, None], 1 - rest_mass)
                probabilities64 = probabilities.double()
                reference = (scores64.gather(1, targets[:, None]).squeeze(1)
                             - (probabilities64 * scores64).sum(1))
                # Normalizing the exact FP32 input separates probability-sum
                # rounding from the GEMM/gather inconsistency.
                normalized_reference = (scores64.gather(1, targets[:, None]).squeeze(1)
                                        - (probabilities64 * scores64).sum(1) / probabilities64.sum(1))
                for precision in ("high", "highest"):
                    torch.set_float32_matmul_precision(precision)
                    actual = helper(probabilities, targets, directions, latents)
                    scores = directions @ latents.T
                    score_centered = (scores.gather(1, targets[:, None]).squeeze(1)
                                      - (probabilities * scores).sum(1))
                    entry = {
                        "rows": rows, "direction": direction_name, "rest_mass": rest_mass,
                        "precision": precision, "device": args.device,
                        "reference_rms": reference.square().mean().sqrt().item(),
                        "probability_sum_max_error": (probabilities64.sum(1) - 1).abs().max().item(),
                        "reference_normalization_error_max": (reference - normalized_reference).abs().max().item(),
                        "mean_embedding": summarize(actual, reference),
                        "score_centered": summarize(score_centered, reference),
                    }
                    results.append(entry)
    report = {
        "scope": "Synthetic policies/directions with actual normalized critic embeddings; numerical audit, not observed collapse causality.",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": checkpoint["step"],
        "source": str(args.source.resolve()), "source_sha256": source_sha,
        "torch": torch.__version__, "device": args.device,
        "cuda_device": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "latent_shape": [vocab, dim], "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(results), "device": args.device}))


if __name__ == "__main__":
    main()

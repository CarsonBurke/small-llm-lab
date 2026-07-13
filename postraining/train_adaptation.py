"""Predictive adaptation: teach the frozen-gate latent-thought model to predict.

Phase 4 of the latent-thought plan.  The gate is masked to EMIT everywhere, so
the closed-loop path is exactly the pretrained model and its BPB stays
comparable against pretraining.  Training adds, on top of continued
pretraining at a reduced backbone learning rate:

- transition beta-NLL replacing the plain latent MSE (same attached mean AND
  attached target paths — the checkpoint architecture is attached-target —
  plus a state-dependent per-dim log-std; reduced as a per-dim mean so its
  gradient scale is a drop-in for the MSE it replaces),
- renderer CE whose gradient may reach the prediction projector but never the
  trunk (the plan's decodability exception),
- paired B=128 SIGReg over token-trajectory embeddings, wired across pairs of
  micro-batches exactly as the PoPE pretraining loop wires
  ``deferred_sigreg_loss``,
- thought exposure: imagined next-token latents inserted at a sampled fraction
  of positions, each predicting the same upcoming real token.

Run after pretraining finishes, e.g.:

    python3 -m postraining.train_adaptation \
        --checkpoint ablation_results/<run>/pretraining_checkpoint.pt \
        --output postraining/runs/<name>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.tensorboard import SummaryWriter

import train_gpt as baseline
from fresh_lejepa_train import FreshHyperparameters
from postraining.adaptation_core import (
    TOKEN_KIND,
    assemble_stream_inputs,
    build_thought_plan,
    copy_last_latent_mse,
    gather_stream_targets,
    weighted_cross_entropy,
)
from postraining.core import JsonlLogger
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import load_model


def build_optimizers(
    wrapper: LatentThoughtModel, backbone_lr_scale: float, new_lr: float
) -> list[torch.optim.Optimizer]:
    """Mirror the pretraining optimizer split at a scaled learning rate."""
    backbone = wrapper.backbone
    hp = FreshHyperparameters
    control_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS + ("delta_c",)
    block_named = list(backbone.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named
        if p.ndim == 2 and not any(pattern in name for pattern in control_patterns)
    ]
    scalar_params = [
        p for name, p in block_named
        if p.ndim < 2 or any(pattern in name for pattern in control_patterns)
    ]
    if len(matrix_params) + len(scalar_params) != len(block_named):
        unrouted = [
            name for name, p in block_named
            if p.ndim > 2 and not any(pattern in name for pattern in control_patterns)
        ]
        raise ValueError(f"parameters fall outside the pretraining split: {unrouted}")
    if backbone.skip_weights.numel() > 0:
        scalar_params.append(backbone.skip_weights)
    token_lr = (hp.tied_embed_lr if hp.tie_embeddings else hp.embed_lr) * backbone_lr_scale
    device_type = backbone.tok_emb.weight.device.type
    adam_kwargs = dict(
        betas=(hp.beta1, hp.beta2), eps=hp.adam_eps, fused=device_type == "cuda"
    )
    optimizer_tok = torch.optim.Adam(
        [{"params": [backbone.tok_emb.weight], "lr": token_lr}], **adam_kwargs
    )
    optimizer_muon = baseline.Muon(
        matrix_params,
        lr=hp.matrix_lr * backbone_lr_scale,
        momentum=hp.muon_momentum,
        backend_steps=hp.muon_backend_steps,
    )
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": hp.scalar_lr * backbone_lr_scale}], **adam_kwargs
    )
    new_params = [p for p in wrapper.new_parameters() if p.requires_grad]
    # Decay-free like every pretraining optimizer; decay would drag the
    # log-std bias toward its ceiling and the zero-init adapter off zero.
    optimizer_new = torch.optim.AdamW(new_params, lr=new_lr, weight_decay=0.0)
    return [optimizer_tok, optimizer_muon, optimizer_scalar, optimizer_new]


def adaptation_losses(
    wrapper: LatentThoughtModel,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    thought_exposure: float,
    thought_ce_weight: float,
    beta: float,
    thought_latent_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    backbone = wrapper.backbone
    trajectory = torch.cat((input_ids, target_ids[:, -1:]), dim=1)
    trajectory_latent = backbone.embed_tokens(trajectory)
    token_latent = trajectory_latent[:, :-1]
    # Deliberately attached, exactly as pretraining: latent-loss gradients
    # enter the target embedding branch too.
    target_latent = trajectory_latent[:, 1:]

    if thought_exposure > 0:
        with torch.no_grad():
            plain_belief = backbone.temporal_belief_from_token_latent(token_latent)
            plain_mean = backbone.prediction_latent(plain_belief)
            thoughts, _ = wrapper.transition.sample(plain_mean, plain_belief)
        insert_mask = torch.rand(token_latent.shape[:2], device=token_latent.device) < thought_exposure
        plan = build_thought_plan(insert_mask, thought_ce_weight, thought_latent_weight)
        thought_inputs = wrapper.adapter(thoughts).to(token_latent.dtype)
        stream_inputs = assemble_stream_inputs(plan, token_latent, thought_inputs)
        stream_targets = gather_stream_targets(plan, target_ids)
        stream_latent_targets = gather_stream_targets(plan, target_latent)
        ce_weights = plan.ce_weight
        latent_weights = plan.latent_weight
        token_slots = (plan.kind == TOKEN_KIND).float()
        thought_fraction = insert_mask.float().mean()
    else:
        stream_inputs = token_latent
        stream_targets = target_ids
        stream_latent_targets = target_latent
        ce_weights = torch.ones(
            token_latent.shape[:2], dtype=torch.float32, device=token_latent.device
        )
        latent_weights = ce_weights
        token_slots = ce_weights
        thought_fraction = torch.zeros((), device=token_latent.device)

    belief = backbone.temporal_belief_from_token_latent(stream_inputs)
    predicted = backbone.prediction_latent(belief)
    latent_nll = wrapper.transition.beta_nll(
        stream_latent_targets, predicted, belief, beta=beta, weights=latent_weights
    )
    # The decodability exception: CE reaches the prediction projector through a
    # detached belief, never the trunk; the input slot is detached as in
    # pretraining.
    ce_predicted = backbone.prediction_latent(belief.detach())
    features = torch.cat((stream_inputs.detach(), ce_predicted), dim=-1)
    logits = backbone.logits_from_features(features)
    renderer_ce = weighted_cross_entropy(logits, stream_targets, ce_weights)

    with torch.no_grad():
        # Diagnostics over token slots only, the same population copy-last
        # covers, so "prediction beats copy-last" compares like with like.
        mask = token_slots.bool()
        prediction_mse = torch.nn.functional.mse_loss(
            predicted.float()[mask], stream_latent_targets.float()[mask]
        )
        log_std = wrapper.transition.log_std(belief[mask])
        copy_last_mse = copy_last_latent_mse(token_latent, target_latent)
    return {
        "latent_nll": latent_nll,
        "renderer_ce": renderer_ce,
        "prediction_mse": prediction_mse,
        "copy_last_mse": copy_last_mse,
        "log_std_mean": log_std.mean(),
        "log_std_floor_fraction": (log_std <= wrapper.transition.log_std_min + 1e-4).float().mean(),
        "thought_fraction": thought_fraction,
    }


class SampledLatentEval(torch.nn.Module):
    """eval_val shim: renderer CE with sampled latents instead of the mean."""

    def __init__(self, wrapper: LatentThoughtModel):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        backbone = self.wrapper.backbone
        token_latent = backbone.embed_tokens(input_ids)
        belief = backbone.temporal_belief_from_token_latent(token_latent)
        mean = backbone.prediction_latent(belief)
        sample, _ = self.wrapper.transition.sample(mean, belief)
        features = torch.cat(
            (token_latent, sample.to(token_latent.dtype)), dim=-1
        )
        logits = backbone.logits_from_features(features)
        return torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1), target_ids.flatten()
        )


def evaluate(
    wrapper: LatentThoughtModel,
    val_tokens: torch.Tensor,
    luts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    grad_accum_steps: int,
) -> dict[str, float]:
    wrapper.eval()
    closed_loss, closed_bpb = baseline.eval_val(
        FreshHyperparameters, wrapper.backbone, 0, 1, device, grad_accum_steps, val_tokens, *luts
    )
    sampled_loss, sampled_bpb = baseline.eval_val(
        FreshHyperparameters, SampledLatentEval(wrapper), 0, 1, device, grad_accum_steps, val_tokens, *luts
    )
    wrapper.train()
    return {
        "val_loss": closed_loss,
        "val_bpb": closed_bpb,
        "sampled_val_loss": sampled_loss,
        "sampled_val_bpb": sampled_bpb,
    }


def save_checkpoint(
    path: Path,
    wrapper: LatentThoughtModel,
    optimizers: list[torch.optim.Optimizer],
    step: int,
    args: argparse.Namespace,
    base_metadata: dict,
    loader: "baseline.DistributedTokenLoader",
) -> None:
    payload = {
        "step": step,
        "model": wrapper.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "args": vars(args),
        "base_metadata": base_metadata,
        "loader": {"file_idx": loader.stream.file_idx, "pos": loader.stream.pos},
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--micro-seqs", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.3)
    parser.add_argument("--new-lr", type=float, default=3e-4)
    parser.add_argument("--new-warmup-steps", type=int, default=50)
    parser.add_argument("--thought-exposure", type=float, default=0.1)
    parser.add_argument("--thought-ce-weight", type=float, default=1.0)
    parser.add_argument("--thought-latent-weight", type=float, default=1.0)
    parser.add_argument("--beta-nll-beta", type=float, default=0.5)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--sigreg-weight", type=float, default=None,
                        help="defaults to the pretraining SIGReg weight")
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--milestones", type=int, nargs="*", default=[1000, 2000])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.grad_accum % 2 != 0:
        parser.error("--grad-accum must be even for paired B=128 SIGReg")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    base_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = load_model(args.checkpoint, device, payload=base_payload)
    pretraining_metadata = (
        base_payload.get("metadata", {}) if isinstance(base_payload, dict) else {}
    )
    # Continue with the checkpoint's own SIGReg weight unless overridden.
    sigreg_weight = args.sigreg_weight
    if sigreg_weight is None:
        sigreg_weight = pretraining_metadata.get("loss", {}).get(
            "sigreg_weight", FreshHyperparameters.sigreg_weight
        )
    base_metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(base_payload.get("step", -1))
        if isinstance(base_payload, dict) else -1,
        "architecture": backbone.architecture,
    }
    del base_payload
    if not hasattr(backbone, "deferred_sigreg_loss"):
        raise ValueError(
            f"{backbone.architecture} has no deferred_sigreg_loss; adaptation "
            "requires a paired-SIGReg (v4/PoPE) checkpoint"
        )
    if 2 * args.micro_seqs != 128:
        print(
            f"warning: 2*micro_seqs={2 * args.micro_seqs} != 128 — the paired "
            "SIGReg statistic no longer matches pretraining's B=128",
            flush=True,
        )
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    wrapper = LatentThoughtModel(backbone).to(device)
    # The gate exists so RL checkpoints have a stable shape, but adaptation
    # never trains or consults it: every position renders.
    for parameter in wrapper.gate.parameters():
        parameter.requires_grad_(False)
    wrapper.train()

    optimizers = build_optimizers(wrapper, args.backbone_lr_scale, args.new_lr)
    # Capture the target LRs before any resume load: a checkpoint written
    # mid-warmup carries warmup-scaled LRs in its param groups.
    base_lrs = [
        [group["lr"] for group in optimizer.param_groups] for optimizer in optimizers
    ]
    loader = baseline.DistributedTokenLoader(FreshHyperparameters.train_files, 0, 1, device)
    start_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        wrapper.load_state_dict(payload["model"], strict=True)
        for optimizer, state in zip(optimizers, payload["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        start_step = int(payload["step"])
        torch.set_rng_state(payload["cpu_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        # Rewind the token stream so a resumed run does not re-train on
        # already-consumed shards.
        stream = loader.stream
        stream.file_idx = int(payload["loader"]["file_idx"])
        stream.tokens = baseline.load_data_shard(stream.files[stream.file_idx])
        stream.pos = int(payload["loader"]["pos"])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output / "metrics.jsonl")
    tensorboard = SummaryWriter(output / "tensorboard")
    manifest = {
        "phase": "predictive_adaptation",
        "args": vars(args),
        "base": base_metadata,
        "sigreg_weight": sigreg_weight,
        "parameters": {
            "backbone": sum(p.numel() for p in backbone.parameters()),
            "new": sum(p.numel() for p in wrapper.new_parameters()),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    tokenizer = spm.SentencePieceProcessor(model_file=FreshHyperparameters.tokenizer_path)
    luts = baseline.build_sentencepiece_luts(
        tokenizer, FreshHyperparameters.vocab_size, device
    )
    seq_len = FreshHyperparameters.train_seq_len
    val_tokens = baseline.load_validation_tokens(FreshHyperparameters.val_files, seq_len)
    global_tokens = args.micro_seqs * seq_len * args.grad_accum

    if start_step == 0 and args.val_every > 0:
        # Pin the pre-adaptation baseline: closed-loop BPB here must match the
        # pretraining checkpoint before any update moves the trunk.
        metrics = evaluate(wrapper, val_tokens, luts, device, args.grad_accum)
        logger.log(type="val", step=0, **metrics)
        for key, value in metrics.items():
            tensorboard.add_scalar(f"adaptation_val/{key}", value, 0)
        print(
            f"step:0/{args.steps} val_bpb:{metrics['val_bpb']:.4f} "
            f"sampled_val_bpb:{metrics['sampled_val_bpb']:.4f}",
            flush=True,
        )

    hp = FreshHyperparameters
    new_optimizer = optimizers[-1]
    muon_optimizer = optimizers[1]
    train_time_ms = 0.0
    for step in range(start_step + 1, args.steps + 1):
        step_started = time.perf_counter()
        warmup = min(step / max(args.new_warmup_steps, 1), 1.0)
        for group, base_lr in zip(new_optimizer.param_groups, base_lrs[-1], strict=True):
            group["lr"] = base_lr * warmup
        # Fresh Muon momentum buffers get the same 0.85->0.95 warmup the
        # pretraining loop uses for its own cold start.
        momentum_frac = (
            min(step / hp.muon_momentum_warmup_steps, 1.0)
            if hp.muon_momentum_warmup_steps > 0 else 1.0
        )
        for group in muon_optimizer.param_groups:
            group["momentum"] = (
                (1 - momentum_frac) * hp.muon_momentum_warmup_start
                + momentum_frac * hp.muon_momentum
            )
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        totals: dict[str, torch.Tensor] = {}
        pending_sigreg_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        for micro_step in range(args.grad_accum):
            input_ids, target_ids = loader.next_batch(global_tokens, seq_len, args.grad_accum)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                losses = adaptation_losses(
                    wrapper,
                    input_ids,
                    target_ids,
                    args.thought_exposure,
                    args.thought_ce_weight,
                    args.beta_nll_beta,
                    args.thought_latent_weight,
                )
            total = (
                losses["renderer_ce"] + args.latent_weight * losses["latent_nll"]
            ) / args.grad_accum
            total.backward()
            # Paired B=128 SIGReg, exactly as the PoPE pretraining loop: one
            # statistic per two micro-batches over trajectory embeddings.
            if micro_step % 2 == 0:
                pending_sigreg_batch = (input_ids, target_ids)
            else:
                assert pending_sigreg_batch is not None
                # Outside autocast, matching the pretraining loop's call site;
                # SIGReg disables autocast internally regardless.
                paired_sigreg = backbone.deferred_sigreg_loss(
                    pending_sigreg_batch[0], pending_sigreg_batch[1], input_ids, target_ids
                )
                (paired_sigreg * sigreg_weight * 2 / args.grad_accum).backward()
                losses["sigreg"] = 2 * paired_sigreg.detach()
            for key, value in losses.items():
                accumulated = totals.get(key)
                contribution = value.detach() / args.grad_accum
                totals[key] = contribution if accumulated is None else accumulated + contribution
        grad_norm = float(
            torch.stack(
                [
                    parameter.grad.detach().float().norm()
                    for parameter in wrapper.parameters()
                    if parameter.grad is not None
                ]
            ).norm()
        )
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        for optimizer in optimizers:
            optimizer.step()
        torch.cuda.synchronize()
        train_time_ms += 1000.0 * (time.perf_counter() - step_started)

        if step % 10 == 0 or step == start_step + 1:
            record = {
                "type": "train",
                "step": step,
                "grad_norm": grad_norm,
                "step_avg_ms": train_time_ms / max(step - start_step, 1),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                **{key: round(float(value), 6) for key, value in totals.items()},
            }
            logger.log(**record)
            for key, value in record.items():
                if isinstance(value, (int, float)) and key != "step":
                    tensorboard.add_scalar(f"adaptation/{key}", value, step)

        if args.val_every > 0 and (step % args.val_every == 0 or step == args.steps):
            metrics = evaluate(wrapper, val_tokens, luts, device, args.grad_accum)
            logger.log(type="val", step=step, **metrics)
            for key, value in metrics.items():
                tensorboard.add_scalar(f"adaptation_val/{key}", value, step)
            print(
                f"step:{step}/{args.steps} val_bpb:{metrics['val_bpb']:.4f} "
                f"sampled_val_bpb:{metrics['sampled_val_bpb']:.4f}",
                flush=True,
            )

        if args.save_every > 0 and (step % args.save_every == 0 or step == args.steps):
            save_checkpoint(
                output / "adaptation_checkpoint.pt",
                wrapper, optimizers, step, args, base_metadata, loader,
            )
        if step in set(args.milestones):
            save_checkpoint(
                output / f"adaptation_step_{step:05d}.pt",
                wrapper, optimizers, step, args, base_metadata, loader,
            )
    tensorboard.close()


if __name__ == "__main__":
    main()

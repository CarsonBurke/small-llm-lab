"""
LeWM-style latent trunk where the detached continuous decoder reads from the same latent that JEPA trains.

Training objective:
- trunk is shaped only by latent prediction MSE + SIGReg
- encoder projector defines the deployed latent state
- detached decoder head predicts a continuous next-token point from that latent
- decoder trains with positive cosine loss to the live token embedding of the target token

Eval path:
- tokens -> trunk hidden -> latent projector -> token head
- decoder output is ranked against the live token embedding table with cosine logits
- BPB is computed from those geometric token scores
"""

from __future__ import annotations

import copy
import io
import math
import os
import random
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train_gpt as base  # noqa: E402


class Hyperparameters(base.Hyperparameters):
    token_metric_weight = float(os.environ.get("TOKEN_METRIC_WEIGHT", "1.0"))
    token_metric_temp = float(os.environ.get("TOKEN_METRIC_TEMP", "0.07"))
    token_metric_expansion = int(os.environ.get("TOKEN_METRIC_EXPANSION", "2"))
    lewm_latent_dim = int(os.environ.get("LEWM_LATENT_DIM", os.environ.get("MODEL_DIM", "512")))
    lewm_pred_weight = float(os.environ.get("LEWM_PRED_WEIGHT", "1.0"))
    lewm_sigreg_weight = float(os.environ.get("LEWM_SIGREG_WEIGHT", "0.1"))
    lewm_pred_expansion = int(os.environ.get("LEWM_PRED_EXPANSION", "2"))
    lewm_pred_dropout = float(os.environ.get("LEWM_PRED_DROPOUT", "0.1"))
    lewm_sigreg_projections = int(os.environ.get("LEWM_SIGREG_PROJECTIONS", "16"))
    lewm_sigreg_stride = int(os.environ.get("LEWM_SIGREG_STRIDE", "128"))
    lewm_sigreg_nodes = int(os.environ.get("LEWM_SIGREG_NODES", "17"))
    lewm_sigreg_tmax = float(os.environ.get("LEWM_SIGREG_TMAX", "4.0"))


class LatentProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = base.CastedLinear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        flat = self.linear(x.reshape(-1, x.size(-1)))
        return flat.reshape(x.size(0), x.size(1), -1)


class LatentPredictor(nn.Module):
    def __init__(self, dim: int, expansion: int, dropout: float):
        super().__init__()
        hidden = max(dim, dim * expansion)
        self.fc = base.CastedLinear(dim, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj = base.CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, latent: Tensor) -> Tensor:
        return latent + self.proj(self.dropout(torch.relu(self.fc(latent))))


class TokenMetricHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, expansion: int):
        super().__init__()
        hidden = max(in_dim, out_dim, in_dim * expansion, out_dim * expansion)
        self.fc = base.CastedLinear(in_dim, hidden, bias=False)
        self.proj = base.CastedLinear(hidden, out_dim, bias=False)
        self.proj._zero_init = True

    def forward(self, latent: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(latent)))


class LeWMSharedLatentMetricGPT(base.GPT):
    training_only_prefixes = ("predictor.", "predictor_projector.")

    def __init__(self, args: Hyperparameters):
        super().__init__(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=True,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
        )
        self.token_metric_weight = args.token_metric_weight
        self.token_metric_temp = args.token_metric_temp
        self.lewm_pred_weight = args.lewm_pred_weight
        self.lewm_sigreg_weight = args.lewm_sigreg_weight
        self.lewm_sigreg_projections = args.lewm_sigreg_projections
        self.lewm_sigreg_stride = args.lewm_sigreg_stride
        self.encoder_projector = LatentProjector(args.model_dim, args.lewm_latent_dim)
        self.predictor = LatentPredictor(args.lewm_latent_dim, args.lewm_pred_expansion, args.lewm_pred_dropout)
        self.predictor_projector = LatentProjector(args.lewm_latent_dim, args.lewm_latent_dim)
        self.token_head = TokenMetricHead(args.lewm_latent_dim, args.model_dim, args.token_metric_expansion)
        nn.init.zeros_(self.predictor.proj.weight)
        nn.init.zeros_(self.token_head.proj.weight)
        t_nodes = torch.linspace(-args.lewm_sigreg_tmax, args.lewm_sigreg_tmax, args.lewm_sigreg_nodes)
        self.register_buffer("sigreg_t_nodes", t_nodes, persistent=False)

    def encode_hidden(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def encode_latent(self, hidden: Tensor) -> Tensor:
        return self.encoder_projector(hidden)

    def predict_next_latent(self, latent: Tensor) -> Tensor:
        return self.predictor_projector(self.predictor(latent))

    def pred_loss(self, latent: Tensor) -> Tensor:
        if latent.size(1) < 2:
            return latent.new_zeros(())
        pred_next = self.predict_next_latent(latent[:, :-1])
        return F.mse_loss(pred_next, latent[:, 1:], reduction="mean")

    def sigreg_loss(self, latent: Tensor) -> Tensor:
        sampled = latent[:, :: self.lewm_sigreg_stride, :].transpose(0, 1).float()
        if sampled.numel() == 0 or sampled.size(1) < 2:
            return latent.new_zeros(())
        dirs = F.normalize(
            torch.randn(self.lewm_sigreg_projections, sampled.size(-1), device=sampled.device, dtype=sampled.dtype),
            dim=-1,
        )
        proj = torch.einsum("sbd,pd->sbp", sampled, dirs)
        t = self.sigreg_t_nodes.to(device=sampled.device, dtype=sampled.dtype)
        tx = proj.unsqueeze(-1) * t
        real = torch.cos(tx).mean(dim=1)
        imag = torch.sin(tx).mean(dim=1)
        target = torch.exp(-0.5 * t.pow(2)).view(1, 1, -1)
        integrand = ((real - target).pow(2) + imag.pow(2)) * target
        return torch.trapezoid(integrand, t, dim=-1).mean()

    def token_point(self, latent: Tensor, *, detach_latent: bool) -> Tensor:
        if detach_latent:
            latent = latent.detach()
        return self.token_head(latent)

    def token_metric_loss(self, latent: Tensor, target_ids: Tensor) -> Tensor:
        pred = F.normalize(self.token_point(latent, detach_latent=True), dim=-1)
        target = F.normalize(self.tok_emb(target_ids), dim=-1)
        return (1.0 - (pred * target).sum(dim=-1)).mean()

    def token_logits(self, latent: Tensor) -> Tensor:
        pred = F.normalize(self.token_point(latent, detach_latent=False), dim=-1)
        protos = F.normalize(self.tok_emb.weight, dim=-1)
        return torch.einsum("btd,vd->btv", pred, protos) / self.token_metric_temp

    def eval_ce_loss(self, latent: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.token_logits(latent)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        hidden = self.encode_hidden(input_ids)
        latent = self.encode_latent(hidden)
        if self.training:
            loss = hidden.new_zeros(())
            if self.lewm_pred_weight > 0.0:
                loss = loss + self.lewm_pred_weight * self.pred_loss(latent)
            if self.lewm_sigreg_weight > 0.0:
                loss = loss + self.lewm_sigreg_weight * self.sigreg_loss(latent)
            if self.token_metric_weight > 0.0:
                loss = loss + self.token_metric_weight * self.token_metric_loss(latent, target_ids)
            return loss
        return self.eval_ce_loss(latent, target_ids)

    def artifact_state_dict(self) -> dict[str, Tensor]:
        return {
            name: tensor
            for name, tensor in self.state_dict().items()
            if not any(name.startswith(prefix) for prefix in self.training_only_prefixes)
        }


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    base.zeropower_via_newtonschulz5 = torch.compile(base.zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = base.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = LeWMSharedLatentMetricGPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, base.CastedLinear):
            module.float()
    base.restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    named_params = list(base_model.named_parameters())
    matrix_params = [
        p
        for name, p in named_params
        if name != "tok_emb.weight" and p.ndim == 2 and not any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in named_params
        if name != "tok_emb.weight" and (p.ndim < 2 or any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS))
    ]

    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = base.Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]

    train_params = sum(p.numel() for p in base_model.parameters())
    artifact_params = sum(t.numel() for t in base_model.artifact_state_dict().values())
    log0(f"model_params_train:{train_params}")
    log0(f"model_params_artifact:{artifact_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"token_metric:weight={args.token_metric_weight} temp={args.token_metric_temp} expansion={args.token_metric_expansion}"
    )
    log0(
        f"lewm_latent:latent_dim={args.lewm_latent_dim} pred_weight={args.lewm_pred_weight} "
        f"sigreg_weight={args.lewm_sigreg_weight} pred_expansion={args.lewm_pred_expansion} "
        f"pred_dropout={args.lewm_pred_dropout} sigreg_projections={args.lewm_sigreg_projections} "
        f"sigreg_stride={args.lewm_sigreg_stride}"
    )
    log0(f"embed_lr:{args.tied_embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    train_loader = base.DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = base.DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = base.eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    artifact_state = base_model.artifact_state_dict()
    if master_process:
        torch.save(artifact_state, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized artifact model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = base.quantize_state_dict_int8(artifact_state)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized artifact int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    roundtrip_state = base.dequantize_state_dict_int8(quant_state)
    missing, unexpected = base_model.load_state_dict(roundtrip_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading roundtrip artifact: {unexpected}")
    if any(not any(name.startswith(prefix) for prefix in base_model.training_only_prefixes) for name in missing):
        raise RuntimeError(f"Unexpected missing keys when loading roundtrip artifact: {missing}")
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = base.eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

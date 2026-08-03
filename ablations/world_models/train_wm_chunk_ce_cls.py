"""
World-model-first chunked language model trained only by downstream token CE.

Architecture:
- encoder maps the full token prefix to token hidden states
- hidden states are pooled into slow chunk states z_k
- a predictor maps previous chunk state z_{k-1} to a predicted current chunk state zhat_k
- a weak lexicalizer reads local token history inside chunk k plus zhat_k to predict chunk-k targets

This keeps the world model on the critical path while avoiding auxiliary losses and detached critic heads.
"""

from __future__ import annotations

import copy
import io
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
    chunk_size = int(os.environ.get("CHUNK_SIZE", "16"))
    wm_latent_dim = int(os.environ.get("WM_LATENT_DIM", str(max(128, int(os.environ.get("MODEL_DIM", "512")) // 2))))
    wm_pred_expansion = int(os.environ.get("WM_PRED_EXPANSION", "2"))
    wm_pred_dropout = float(os.environ.get("WM_PRED_DROPOUT", "0.1"))
    decoder_layers = int(os.environ.get("DECODER_LAYERS", "1"))
    decoder_heads = int(os.environ.get("DECODER_HEADS", "8"))
    decoder_kv_heads = int(os.environ.get("DECODER_KV_HEADS", "4"))
    decoder_mlp_mult = int(os.environ.get("DECODER_MLP_MULT", "2"))
    lex_embed_lr = float(os.environ.get("LEX_EMBED_LR", str(base.Hyperparameters.head_lr)))


class LatentProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = base.CastedLinear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        flat = self.linear(x.reshape(-1, x.size(-1)))
        return flat.reshape(x.size(0), x.size(1), -1)


class ChunkCLSPool(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(model_dim, dtype=torch.float32) * 0.02)

    def forward(self, hidden: Tensor) -> Tensor:
        bsz, num_chunks, chunk_size, dim = hidden.shape
        q = self.query.to(dtype=hidden.dtype)[None, None, None, :]
        attn = (hidden * q).sum(dim=-1) / (dim ** 0.5)
        weights = attn.softmax(dim=-1)
        return (weights[..., None] * hidden).sum(dim=2)


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


class ChunkLexicalDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        logit_softcap: float,
    ):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.latent_proj = base.CastedLinear(latent_dim, model_dim, bias=False)
        self.blocks = nn.ModuleList(
            [
                base.Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = base.RMSNorm()
        self.lm_head = base.CastedLinear(model_dim, vocab_size, bias=False)

    def forward(self, input_ids: Tensor, cond_latent: Tensor) -> Tensor:
        bsz, num_chunks, chunk_size = input_ids.shape
        x = self.tok_emb(input_ids.reshape(bsz * num_chunks, chunk_size))
        cond = self.latent_proj(cond_latent.reshape(bsz * num_chunks, cond_latent.size(-1))).unsqueeze(1)
        x = x + cond
        x0 = x
        for block in self.blocks:
            x = block(x, x0)
        x = self.final_norm(x)
        logits_proj = self.lm_head(x.reshape(-1, x.size(-1)))
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits.reshape(bsz, num_chunks, chunk_size, -1)


class WMChunkCEGPT(base.GPT):
    training_only_prefixes: tuple[str, ...] = ()

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
        if args.train_seq_len % args.chunk_size != 0:
            raise ValueError(f"TRAIN_SEQ_LEN={args.train_seq_len} must be divisible by CHUNK_SIZE={args.chunk_size}")
        if args.decoder_heads > args.model_dim or args.model_dim % args.decoder_heads != 0:
            raise ValueError("decoder_heads must divide model_dim")
        self.chunk_size = args.chunk_size
        self.chunk_pool = ChunkCLSPool(args.model_dim)
        self.encoder_projector = LatentProjector(args.model_dim, args.wm_latent_dim)
        self.predictor = LatentPredictor(args.wm_latent_dim, args.wm_pred_expansion, args.wm_pred_dropout)
        self.predictor_projector = LatentProjector(args.wm_latent_dim, args.wm_latent_dim)
        self.lexical_decoder = ChunkLexicalDecoder(
            vocab_size=args.vocab_size,
            model_dim=args.model_dim,
            latent_dim=args.wm_latent_dim,
            num_layers=args.decoder_layers,
            num_heads=args.decoder_heads,
            num_kv_heads=args.decoder_kv_heads,
            mlp_mult=args.decoder_mlp_mult,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            logit_softcap=args.logit_softcap,
        )
        self.init_latent = nn.Parameter(torch.zeros(args.wm_latent_dim, dtype=torch.float32))
        nn.init.zeros_(self.predictor.proj.weight)

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

    def pool_chunks(self, hidden: Tensor) -> Tensor:
        bsz, seqlen, dim = hidden.shape
        num_chunks = seqlen // self.chunk_size
        hidden = hidden.reshape(bsz, num_chunks, self.chunk_size, dim)
        return self.chunk_pool(hidden)

    def encode_latent(self, hidden: Tensor) -> Tensor:
        return self.encoder_projector(self.pool_chunks(hidden))

    def predict_chunk_latents(self, latent: Tensor) -> Tensor:
        init = self.init_latent.to(dtype=latent.dtype)[None, None, :].expand(latent.size(0), 1, -1)
        prev = torch.cat((init, latent[:, :-1, :]), dim=1)
        return self.predictor_projector(self.predictor(prev))

    def chunk_ce_loss(self, input_ids: Tensor, target_ids: Tensor, cond_latent: Tensor) -> Tensor:
        bsz, seqlen = input_ids.shape
        num_chunks = seqlen // self.chunk_size
        input_chunks = input_ids.reshape(bsz, num_chunks, self.chunk_size)
        target_chunks = target_ids.reshape(bsz, num_chunks, self.chunk_size)
        logits = self.lexical_decoder(input_chunks, cond_latent)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_chunks.reshape(-1), reduction="mean")

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        hidden = self.encode_hidden(input_ids)
        latent = self.encode_latent(hidden)
        pred_latent = self.predict_chunk_latents(latent)
        return self.chunk_ce_loss(input_ids, target_ids, pred_latent)

    def artifact_state_dict(self) -> dict[str, Tensor]:
        return {name: tensor for name, tensor in self.state_dict().items()}


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

    base_model = WMChunkCEGPT(args).to(device).bfloat16()
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
        if name not in {"tok_emb.weight", "lexical_decoder.tok_emb.weight"}
        and p.ndim == 2
        and not any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in named_params
        if name not in {"tok_emb.weight", "lexical_decoder.tok_emb.weight"}
        and (p.ndim < 2 or any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS))
    ]

    optimizer_enc_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_lex_tok = torch.optim.Adam(
        [{"params": [base_model.lexical_decoder.tok_emb.weight], "lr": args.lex_embed_lr, "base_lr": args.lex_embed_lr}],
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
    optimizers: list[torch.optim.Optimizer] = [optimizer_enc_tok, optimizer_lex_tok, optimizer_muon, optimizer_scalar]

    train_params = sum(p.numel() for p in base_model.parameters())
    artifact_params = sum(t.numel() for t in base_model.artifact_state_dict().values())
    log0(f"model_params_train:{train_params}")
    log0(f"model_params_artifact:{artifact_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"wm_chunk_ce:chunk_size={args.chunk_size} latent_dim={args.wm_latent_dim} "
        f"pool=cls "
        f"pred_expansion={args.wm_pred_expansion} pred_dropout={args.wm_pred_dropout} "
        f"decoder_layers={args.decoder_layers} decoder_heads={args.decoder_heads} "
        f"decoder_kv_heads={args.decoder_kv_heads} decoder_mlp_mult={args.decoder_mlp_mult}"
    )
    log0(
        f"enc_embed_lr:{args.tied_embed_lr} lex_embed_lr:{args.lex_embed_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
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
